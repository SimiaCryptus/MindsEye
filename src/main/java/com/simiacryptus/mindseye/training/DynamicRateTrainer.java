package com.simiacryptus.mindseye.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DynamicRateTrainer extends DelegateTrainer<RateTrainingComponent> {
  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);

  public double alpha = 2.;
  public double beta= 3.;
  public double end = 1e-9;
  public double start = 0.1;
  private double max = 10000;
  private boolean verbose = false;

  protected DynamicRateTrainer() {
    this(new GradientDescentTrainer());
  }

  public DynamicRateTrainer(final RateTrainingComponent inner) {
    super(inner);
  }

  public double getMax() {
    return this.max;
  }

  public TrainingComponent setMaxRate(final double maxRate) {
    this.max = maxRate;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) {
    double prevError = getError();
    double x = start;
    while (!Double.isFinite(this.inner.getError()) || this.inner.getError() > trainingContext.terminalErr) {
      this.inner.setRate(toRate(x));
      final TrainingStep step = this.inner.step(trainingContext);
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
      if (step.getStartError() < step.testError) {
        x /= Math.pow(alpha, beta);
      } else {
        if (x > this.max) {
          if (isVerbose()) {
            log.debug(String.format("rate overflow: %s", x));
            // break;
          }
        } else {
          x *= alpha;
        }
      }
      if (x < end) {
        if (isVerbose()) {
          log.debug(String.format("TERMINAL rate underflow: %s", x));
        }
        break;
      }
    }
    if (isVerbose()) {
      String string = getNet().toString();
      if (string.length() > 1024) {
        string = string.substring(0, 1924);
      }
      DynamicRateTrainer.log.debug("Final network state: " + string);
    }
    final double endError = getError();
    return new TrainingStep(prevError, endError, true);
  }

  private double toRate(double x) {
    return x*x;
  }
}
