package com.simiacryptus.mindseye.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class IterativeTrainer extends DelegateTrainer<TrainingComponent> {
  private static final Logger log = LoggerFactory.getLogger(IterativeTrainer.class);
  private int stepCounter = 0;

  protected IterativeTrainer() {
    this(new LineSearchTrainer());
  }

  public IterativeTrainer(final TrainingComponent inner) {
    super(inner);
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) {
    double prevError = getError();
    while (!Double.isFinite(this.inner.getError()) || this.inner.getError() > trainingContext.terminalErr) {
      long start = System.nanoTime();
      final TrainingStep step = this.inner.step(trainingContext);
      double elapsed = (System.nanoTime() - start) / 1000000000.0;
      onStep(new StepState(stepCounter++, step.finalError(), elapsed));
      if (!Double.isFinite(prevError)) {
        prevError = step.getStartError();
      }
      if (trainingContext.timeout < System.currentTimeMillis()) {
        log.debug(String.format("TIMEOUT; current err: %s", step));
        break;
      }
      if (step.finalError() <= trainingContext.terminalErr) {
        if (isVerbose()) {
          log.debug(String.format("TERMINAL Final err: %s", step));
        }
      }
    }
    if (isVerbose()) {
      String string = getNet().toString();
      if (string.length() > 1024) {
        string = string.substring(0, 1924);
      }
      IterativeTrainer.log.debug("Final network state: " + string);
    }
    final double endError = getError();
    return new TrainingStep(prevError, endError, true);
  }

  public final ArrayList<StepState> history = new ArrayList<>();

  protected void onStep(StepState stepState) {
    history.add(stepState);
  }

  public static class StepState {
    private final int iteration;
    private final double fitness;
    private final double evaluationTime;

    private StepState(int evaluations, double fitness, double elapsed) {
      iteration = evaluations;
      this.fitness = fitness;
      evaluationTime = elapsed;
    }

    public int getIteration() {
      return iteration;
    }

    public double getFitness() {
      return fitness;
    }

    public double getEvaluationTime() {
      return evaluationTime;
    }
  }
}
