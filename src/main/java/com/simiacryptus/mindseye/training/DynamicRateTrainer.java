package com.simiacryptus.mindseye.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.TrainingContext;
import com.simiacryptus.mindseye.net.DAGNetwork;

public class DynamicRateTrainer implements TrainingComponent {
  private static class UniformAdaptiveRateParams {
    public final double alpha;
    public final double beta;
    public final double endRate;
    public final double startRate;

    public UniformAdaptiveRateParams(final double startRate, final double endRate, final double alpha, final double beta) {
      this.endRate = endRate;
      this.alpha = alpha;
      this.beta = beta;
      this.startRate = startRate;
    }
  }

  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);

  private long etaSec = Long.MAX_VALUE;
  private final RateTrainingComponent inner;
  private double maxRate = 10000;
  private double minRate = 0;
  private boolean verbose = false;

  protected DynamicRateTrainer() {
    super();
    this.inner = new GradientDescentTrainer();
  }

  public DynamicRateTrainer(final RateTrainingComponent inner) {
    super();
    this.inner = inner;
  }

  @Override
  public NDArray[][] getData() {
    return this.inner.getData();
  }

  @Override
  public double getError() {
    return this.inner.getError();
  }

  public long getEtaMs() {
    return this.etaSec;
  }

  public double getMaxRate() {
    return this.maxRate;
  }

  public double getMinRate() {
    return this.minRate;
  }

  @Override
  public DAGNetwork getNet() {
    return this.inner.getNet();
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  @Override
  public void reset() {
    this.inner.reset();
  }

  @Override
  public TrainingComponent setData(final NDArray[][] data) {
    return this.inner.setData(data);
  }

  public TrainingComponent setEtaEnd(final long cnt, final java.util.concurrent.TimeUnit units) {
    this.etaSec = units.toSeconds(cnt);
    return this;
  }

  public TrainingComponent setMaxRate(final double maxRate) {
    this.maxRate = maxRate;
    return this;
  }

  public TrainingComponent setMinRate(final double minRate) {
    this.minRate = minRate;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) {
    return train(trainingContext, new UniformAdaptiveRateParams(0.1, 1e-9, 2., 3.));
  }

  private TrainingStep train(final TrainingContext trainingContext, final UniformAdaptiveRateParams params) {
    double prevError = getError();
    final TrainingComponent gradientDescentTrainer = this.inner;
    double rate = params.startRate;
    while (!Double.isFinite(gradientDescentTrainer.getError()) || gradientDescentTrainer.getError() > trainingContext.terminalErr) {
      this.inner.setRate(rate);
      final TrainingStep step = gradientDescentTrainer.step(trainingContext);
      if (!Double.isFinite(prevError)) {
        prevError = step.getStartError();
      }
      if (trainingContext.timeout < System.currentTimeMillis()) {
        log.debug(String.format("TIMEOUT; current err: %s", step));
        break;
      }
      if (step.finalError() <= trainingContext.terminalErr) {
        if (isVerbose()) {
        }
        log.debug(String.format("TERMINAL Final err: %s", step));
      }
      if (step.getStartError() < step.testError) {
        rate /= Math.pow(params.alpha, params.beta);
      } else {
        if (rate > this.maxRate) {
          if (isVerbose()) {
            log.debug(String.format("rate overflow: %s", rate));
            // break;
          }
        } else {
          rate *= params.alpha;
        }
      }
      if (rate < params.endRate) {
        if (isVerbose()) {
        }
        log.debug(String.format("TERMINAL rate underflow: %s", rate));
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
}
