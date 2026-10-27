//! Warmup + cosine learning-rate schedule: the default (tests, single steps) and the schedule of a training run,
//! which reaches its floor with the last step of the run.
#[cfg(test)]
mod test_learning_rate_schedule {
    use crate::neural_networks::utils::adam_w::{LearningRateSchedule, DEFAULT_SCHEDULE_STEPS, MIN_LR_RATIO, WARMUP_STEPS};

    const BASE_LR: f64 = 0.001;

    fn assert_lr(schedule: &LearningRateSchedule, step: usize, expected: f64) {
        let lr = schedule.learning_rate(BASE_LR, step);
        assert!((lr - expected).abs() < 1e-12, "step {}: learning rate {} != {} ({:?})", step, lr, expected, schedule);
    }

    #[test]
    fn test_default_schedule_warms_up_and_reaches_the_floor_after_2000_steps() {
        let schedule = LearningRateSchedule::DEFAULT;
        assert_lr(&schedule, 0, 0.0);
        assert_lr(&schedule, WARMUP_STEPS / 2, BASE_LR / 2.0);
        assert_lr(&schedule, WARMUP_STEPS, BASE_LR);
        assert_lr(&schedule, DEFAULT_SCHEDULE_STEPS, MIN_LR_RATIO * BASE_LR);
        assert_lr(&schedule, 10 * DEFAULT_SCHEDULE_STEPS, MIN_LR_RATIO * BASE_LR);
    }

    /// 300 epochs x 67 batches (1071 records, batch size 16): the floor is reached with the last batch, not after 2000 steps.
    #[test]
    fn test_run_schedule_spans_all_epochs() {
        let run_steps = 300 * 67;
        let schedule = LearningRateSchedule::for_run(0, run_steps);
        assert_eq!(schedule.warmup_steps, WARMUP_STEPS);

        assert_lr(&schedule, WARMUP_STEPS, BASE_LR);
        let halfway = WARMUP_STEPS + (run_steps - WARMUP_STEPS) / 2;
        assert_lr(&schedule, halfway, BASE_LR * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5));
        assert!(schedule.learning_rate(BASE_LR, DEFAULT_SCHEDULE_STEPS) > 0.9 * BASE_LR, "after 2000 steps the run is still at the start");
        assert_lr(&schedule, run_steps, MIN_LR_RATIO * BASE_LR);

        let rates: Vec<f64> = (WARMUP_STEPS..=run_steps).map(|step| schedule.learning_rate(BASE_LR, step)).collect();
        assert!(rates.windows(2).all(|pair| pair[1] <= pair[0]), "the learning rate must only decrease after the warmup");
    }

    /// Continuing a saved model (time step 5000) starts a new warmup + cosine over the new run.
    #[test]
    fn test_resumed_run_starts_a_new_schedule() {
        let schedule = LearningRateSchedule::for_run(5000, 1000);
        assert_eq!(schedule.warmup_steps, 100, "the warmup takes at most 10 % of the run");
        assert_lr(&schedule, 5000 + 50, BASE_LR / 2.0);
        assert_lr(&schedule, 5000 + 100, BASE_LR);
        assert_lr(&schedule, 5000 + 1000, MIN_LR_RATIO * BASE_LR);
    }

    /// A run shorter than 10 steps has no warmup and still ends at the floor.
    #[test]
    fn test_short_run_has_no_warmup() {
        let schedule = LearningRateSchedule::for_run(0, 5);
        assert_eq!(schedule.warmup_steps, 0);
        assert!(schedule.learning_rate(BASE_LR, 1) > 0.5 * BASE_LR);
        assert_lr(&schedule, 5, MIN_LR_RATIO * BASE_LR);
    }
}
