package com.simiacryptus.mindseye.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;

public class DynamicRateTrainer implements TrainingComponent {
  private static class UniformAdaptiveRateParams {
    public final double alpha;
    public final double beta;
    public final double endRate;
    public final double startRate;
    public final double terminalETA;

    public UniformAdaptiveRateParams(final double startRate, final double endRate, final double alpha, final double beta, final double terminalETA) {
      this.endRate = endRate;
      this.alpha = alpha;
      this.beta = beta;
      this.startRate = startRate;
      this.terminalETA = terminalETA;
    }
  }

  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);

  int currentIteration = 0;
  private long etaSec = java.util.concurrent.TimeUnit.HOURS.toSeconds(1);
  private int evolutionPhases = 2;
  int generationsSinceImprovement = 0;
  private final RateTrainingComponent inner;
  int lastCalibratedIteration = Integer.MIN_VALUE;
  int maxIterations = 100;
  private double maxRate = 10000;
  private double minRate = 0;
  private boolean verbose = false;
  
  public DynamicRateTrainer(RateTrainingComponent inner) {
    super();
    this.inner = inner;
  }

  protected DynamicRateTrainer() {
    super();
    inner = new GradientDescentTrainer();
  }

  private boolean evolve(final TrainingContext trainingContext) {
    final boolean isValid = null != this.inner.getNet().evolve();
    if (isValid) {
      this.inner.refresh();
      this.inner.step(trainingContext);
    }
    return isValid;
  }

  @Override
  public double getError() {
    return this.inner.getError();
  }

  public long getEtaMs() {
    return this.etaSec;
  }

  public int getEvolutionPhases() {
    return this.evolutionPhases;
  }

  public int getGenerationsSinceImprovement() {
    return this.generationsSinceImprovement;
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

  public TrainingComponent setEtaEnd(final long cnt, final java.util.concurrent.TimeUnit units) {
    this.etaSec = units.toSeconds(cnt);
    return this;
  }

  public TrainingComponent setEvolutionPhases(final int evolutionPhases) {
    this.evolutionPhases = evolutionPhases;
    return this;
  }

  public TrainingComponent setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    this.generationsSinceImprovement = generationsSinceImprovement;
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
  public double step(final TrainingContext trainingContext) {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    int lifecycle = 0;
    do {
      train(trainingContext, new UniformAdaptiveRateParams(0.1, 1e-9, 1.2, 2., getEtaMs()));
    } while (lifecycle++ < getEvolutionPhases() && evolve(trainingContext));
    // train2(trainingContext);
    return getError();
  }

  private void train(final TrainingContext trainingContext, final UniformAdaptiveRateParams params) {
    final TrainingComponent gradientDescentTrainer = this.inner;
    double rate = params.startRate;
    final RateMonitor linearLearningRate = new RateMonitor(5000);
    while (!Double.isFinite(gradientDescentTrainer.getError()) || gradientDescentTrainer.getError() > trainingContext.terminalErr) {
      this.inner.setRate(rate);
      final double delta = gradientDescentTrainer.step(trainingContext);
      final double error = gradientDescentTrainer.getError();
      final double rateDelta = linearLearningRate.add(delta);
      final double projectedEndSeconds = -error / (rateDelta * 1000.);
      if (isVerbose()) {
        log.debug(String.format("Projected final convergence time: %.3f sec; %s - %s/sec", projectedEndSeconds, error, rateDelta));
      }
      if (trainingContext.timeout < System.currentTimeMillis()) {
        log.debug(String.format("TIMEOUT; current err: %s", error));
        break;
      }
      if (Double.isFinite(projectedEndSeconds) && projectedEndSeconds > params.terminalETA) {
        log.debug(String.format("TERMINAL Projected final convergence time: %.3f sec", projectedEndSeconds));
        break;
      }
      if (error <= trainingContext.terminalErr) {
        if (isVerbose()) {
        }
        log.debug(String.format("TERMINAL Final err: %s", error));
      }
      if (0. <= delta) {
        rate /= Math.pow(params.alpha, params.beta);
      } else if (0. > delta) {
        rate *= params.alpha;
      } else {
        assert false;
      }
      if (rate < params.endRate) {
        if (isVerbose()) {
        }
        log.debug(String.format("TERMINAL rate underflow: %s", rate));
        break;
      }
    }
    if (isVerbose()) {
      DynamicRateTrainer.log.debug("Final network state: " + this.inner.getNet().toString());
    }
  }

  public NDArray[][] getData() {
    return inner.getData();
  }

  @Override
  public void refresh() {
    inner.refresh();
  }
}
