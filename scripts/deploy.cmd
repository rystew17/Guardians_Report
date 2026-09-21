@echo off
REM Put the app on Cloud Run, with the data bucket mounted. Windows twin of
REM deploy.sh -- same flags, same .env, so the two cannot drift into deploying
REM different things.
REM
REM Run it from anywhere; it finds the project root itself:
REM
REM   scripts\deploy.cmd
REM
REM It needs the gcloud CLI logged in, which is a separate credential from the
REM application-default one the Python client uses -- a working sync_data.py
REM does not mean this will work:
REM
REM   gcloud auth login
REM   gcloud config set project <your project>
REM
REM The flags that are not obvious:
REM
REM   --source .           builds with Cloud Build, so no local Docker needed
REM   --add-volume/--mount the data bucket appears at /gcs, which is what makes
REM                        the corpus readable and the odds record durable --
REM                        Cloud Run's own filesystem does not survive the
REM                        request that wrote to it
REM   --memory 4Gi          measured, not guessed. The build process itself peaks
REM                         near 440MB, but files read through the GCS mount are
REM                         cached by the kernel, and that cache counts against the
REM                         container's limit -- so memory climbs for the whole run
REM                         rather than settling. At 2Gi it crossed the limit about
REM                         fifteen minutes in and the container was killed, twice.
REM                         That reaches the phone only as "lost connection": the
REM                         kill takes down the event stream that would have
REM                         carried the reason.
REM   --timeout 3600       the platform cap, and not optional here. Cloud Run ends
REM                        every request at this limit, the build's event stream
REM                        included -- and when that stream ends with nothing else
REM                        in flight the instance is reclaimed and the build inside
REM                        it is killed. At the 900s default every build died at
REM                        almost exactly fifteen minutes, whatever else was fixed.
REM   --max-instances 1    one instance, because the job table lives in that
REM                        process's memory. At two, a POST to /api/generate
REM                        could create the job on one instance and the
REM                        /api/events stream for it land on the other, which
REM                        has never heard of that job id -- so the stream ends
REM                        at once and the phone shows a failure while the
REM                        build it started runs happily on the other instance.
REM                        That is the "first attempt always fails, second
REM                        always works" bug: after the first request an
REM                        instance is warm, so both halves land together.
REM                        Session affinity is best-effort and would only make
REM                        it rarer. One instance removes the class.
REM   --min-instances 0    scale to nothing when idle; a cold start costs a
REM                        slower first request and no money in between
REM   --no-cpu-throttling  the one that is not optional. Cloud Run allocates CPU
REM                        only while a request is in flight, and a report build
REM                        runs after the response that started it -- throttled,
REM                        it logged one line and then sat for ten minutes.

setlocal enabledelayedexpansion
cd /d "%~dp0.."

if not defined SERVICE set "SERVICE=guards-report"
if not defined REGION set "REGION=us-central1"

if not exist ".env" (
  echo Cannot find .env -- is this the project root?
  exit /b 1
)

if not defined PROJECT (
  for /f "usebackq delims=" %%P in (`gcloud config get-value project 2^>nul`) do set "PROJECT=%%P"
)
if "%PROJECT%"=="(unset)" set "PROJECT="

REM Pull the rest straight out of .env so this and deploy.sh stay identical.
REM GCP_PROJECT is read here too, as a fallback for a gcloud config that has no
REM project set. That is not hypothetical: `configurations/config_default` on
REM this machine is a zero-byte file, so `gcloud config get-value project`
REM answers with nothing and the deploy stopped before it started -- while the
REM project id sat in .env the whole time, which is where every other setting
REM comes from anyway.
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if /i "%%A"=="GCP_PROJECT"  if not defined PROJECT  set "PROJECT=%%B"
  if /i "%%A"=="GCS_BUCKET"   if not defined BUCKET   set "BUCKET=%%B"
  if /i "%%A"=="ODDS_API_KEY" if not defined ODDS_KEY set "ODDS_KEY=%%B"
  REM Read before generating. A fresh token on every deploy is a redeploy that
  REM silently invalidates the bookmark on your phone, which reads as the
  REM service having broken rather than as the token having changed.
  if /i "%%A"=="ACCESS_TOKEN" if not defined TOKEN    set "TOKEN=%%B"
)

if not defined PROJECT goto :noproject
if "%PROJECT%"=="(unset)" goto :noproject
if not defined BUCKET (
  echo No GCS_BUCKET found in .env
  exit /b 1
)

if not defined TOKEN (
  REM Generated rather than defaulted: a deployment with a guessable token is a
  REM deployment with none, and a blank one leaves the service wide open.
  for /f "usebackq delims=" %%T in (`".venv\Scripts\python.exe" -c "import secrets; print(secrets.token_urlsafe(24))"`) do set "TOKEN=%%T"
  echo Generated an access token. Save it -- it is the only thing between the
  echo URL and your odds quota:
  echo.
  echo     !TOKEN!
  echo.
)

echo Deploying %SERVICE% to %REGION% in %PROJECT%, bucket %BUCKET% ...

call gcloud run deploy "%SERVICE%" ^
  --source . ^
  --project "%PROJECT%" ^
  --region "%REGION%" ^
  --platform managed ^
  --allow-unauthenticated ^
  --no-cpu-throttling ^
  --memory 4Gi ^
  --cpu 2 ^
  --timeout 3600 ^
  --min-instances 0 ^
  --max-instances 1 ^
  --concurrency 4 ^
  --add-volume "name=data,type=cloud-storage,bucket=%BUCKET%" ^
  --add-volume-mount "volume=data,mount-path=/gcs" ^
  --set-env-vars "GCP_PROJECT=%PROJECT%,GCS_BUCKET=%BUCKET%,DATA_DIR=/gcs/data,RAW_ARCHIVE_DIR=/gcs/data/raw,OUTPUT_DIR=/gcs/out,ODDS_API_KEY=%ODDS_KEY%,ACCESS_TOKEN=%TOKEN%"

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Deploy failed with exit code %ERRORLEVEL%.
  exit /b %ERRORLEVEL%
)

for /f "usebackq delims=" %%U in (`gcloud run services describe "%SERVICE%" --project "%PROJECT%" --region "%REGION%" --format^="value(status.url)"`) do set "URL=%%U"

echo.
echo Deployed. Bookmark this on your phone -- the token is in the link, and
echo the app sets a cookie from it so the rest of the site works without it:
echo.
echo     !URL!/?k=!TOKEN!
echo.
exit /b 0

:noproject
echo No project found. Either set one for gcloud:
echo.
echo     gcloud config set project ^<id^>
echo.
echo or put GCP_PROJECT=^<id^> in .env, which this script reads as a fallback.
exit /b 1
