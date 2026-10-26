@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===== 1/3 Check data layout =====
python ms_train.py --check-only
if errorlevel 1 goto :err
echo ===== 2/3 Train (5 seeds x 9 features; can be interrupted and resumed) =====
python ms_train.py
if errorlevel 1 goto :err
echo ===== 3/3 Predict on the internal and external cohorts =====
python ms_predict.py --original
if errorlevel 1 goto :err
echo Done. Summarize with: python ms_aggregate.py results
pause
exit /b 0
:err
echo Error: see runs\train_log.txt or results\predict_log.txt
pause
exit /b 1
