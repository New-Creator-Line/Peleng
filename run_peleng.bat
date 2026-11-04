@echo off
chcp 65001 > nul
echo ============================================================
echo   PELENG 3D VISION & TRACKING ENGINE [10D CTRV UKF]
echo ============================================================
echo.
echo Запуск 3D реконструкции и фильтрации траектории...
python main.py --tracker-type ukf %*
echo.
echo ============================================================
echo Обработка успешно завершена.
echo Результаты: results\trajectories\output_trajectory_ukf.csv
echo Метрики:    results\metrics\metrics_summary_ukf.json
echo Графики:    results\plots\
echo ============================================================
pause
