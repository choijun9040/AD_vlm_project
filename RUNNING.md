```bash
# train_teacher
nohup python scripts/train_teacher.py > logs/train_teacher.log 2>&1 & echo $! > logs/train_teacher.pid
tail -f logs/train_teacher.log
cat logs/train_teacher.pid
kill -0 $(cat logs/train_teacher.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/train_teacher.pid)

# generate_hazard_labels
python scripts/generate_hazard_labels.py 2>&1 | tee logs/hazard_labels.log

# train_distillation (hazard percentile 가중치 + 오버샘플링 적용 버전 -> output_dir도 student_distill_2)
nohup python scripts/train_distillation.py > logs/distillation_2.log 2>&1 & echo $! > logs/distillation_2.pid
tail -f logs/distillation_2.log
cat logs/distillation_2.pid
kill -0 $(cat logs/distillation_2.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/distillation_2.pid)

# train_baseline (오버샘플링 적용 버전 -> output_dir도 student_baseline_2)
nohup python scripts/train_baseline.py > logs/baseline_2.log 2>&1 & echo $! > logs/baseline_2.pid
tail -f logs/baseline_2.log
cat logs/baseline_2.pid
kill -0 $(cat logs/baseline_2.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/baseline_2.pid)

# train_kd_only.py (오버샘플링 적용 버전 -> output_dir도 student_kd_only_2)
nohup python scripts/train_kd_only.py > logs/kd_only_2.log 2>&1 & echo $! > logs/kd_only_2.pid
tail -f logs/kd_only_2.log
cat logs/kd_only_2.pid
kill -0 $(cat logs/kd_only_2.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/kd_only_2.pid)

# tmux 마우스 스크롤
vim ~/.tmux.conf
set -g mouse on
tmux source ~/.tmux.conf
```
