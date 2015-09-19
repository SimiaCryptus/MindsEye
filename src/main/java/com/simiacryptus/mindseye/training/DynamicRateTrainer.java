package com.simiacryptus.mindseye.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;

public class DynamicRateTrainer implements TrainingComponent {
  private static class UniformAdaptiveRateParams {
    public final double alpha;
    public final double beta;
    public final double endRate;
    public final double startRate;
    public final double terminalETA;
    public final double terminalErr;

    public UniformAdaptiveRateParams(final double startRate, final double endRate, final double alpha, final double beta, final double convergence, final double terminalETA) {
      this.endRate = endRate;
      this.alpha = alpha;
      this.beta = beta;
      this.startRate = startRate;
      this.terminalErr = convergence;
      this.terminalETA = terminalETA;
    }
  }

  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);

  int currentIteration = 0;
  private long etaSec = java.util.concurrent.TimeUnit.HOURS.toSeconds(1);
  int generationsSinceImprovement = 0;
  private final GradientDescentTrainer inner = new GradientDescentTrainer();
  int lastCalibratedIteration = Integer.MIN_VALUE;
  int maxIterations = 100;
  private double maxRate = 10000;
  private double minRate = 0;
  private double stopError = 1e-2;
  private boolean verbose = false;

  public long getEtaMs() {
    return this.etaSec;
  }

  public int getGenerationsSinceImprovement() {
    return this.generationsSinceImprovement;
  }

  public GradientDescentTrainer getGradientDescentTrainer() {
    return this.inner;
  }

  public double getMaxRate() {
    return this.maxRate;
  }

  public double getMinRate() {
    return this.minRate;
  }

  public double getStopError() {
    return this.stopError;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public TrainingComponent setEtaEnd(final long cnt, final java.util.concurrent.TimeUnit units) {
    this.etaSec = units.toSeconds(cnt);
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

  public TrainingComponent setStopError(final double stopError) {
    this.stopError = stopError;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    this.verbose = verbose;
    getGradientDescentTrainer().setVerbose(verbose);
    return this;
  }
  
  private int evolutionPhases = 2;

  @Override
  public double step(final TrainingContext trainingContext) {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    int lifecycle = 0;
    do {
      train(trainingContext, new UniformAdaptiveRateParams(0.1, 1e-9, 1.2, 2., this.stopError, getEtaMs()));
    } while (lifecycle++ < getEvolutionPhases() && evolve(trainingContext));
    // train2(trainingContext);
    return getError();
  }

  private boolean evolve(TrainingContext trainingContext) {
    boolean isValid = null != getGradientDescentTrainer().getNet().evolve();
    if(isValid) {
      getGradientDescentTrainer().setRate(0);
      getGradientDescentTrainer().step(trainingContext);
    }
    return isValid;
  }

  private void train(final TrainingContext trainingContext, final UniformAdaptiveRateParams params) {
    final TrainingComponent gradientDescentTrainer = getGradientDescentTrainer();
    double rate = params.startRate;
    final RateMonitor linearLearningRate = new RateMonitor(5000);
    while (!Double.isFinite(gradientDescentTrainer.getError()) || gradientDescentTrainer.getError() > params.terminalErr) {
      getGradientDescentTrainer().setRate(rate);
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
      if (error <= params.terminalErr) {
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
      DynamicRateTrainer.log.debug("Final network state: " + getGradientDescentTrainer().getNet().toString());
    }
  }

  public int getEvolutionPhases() {
    return evolutionPhases;
  }

  public TrainingComponent setEvolutionPhases(int evolutionPhases) {
    this.evolutionPhases = evolutionPhases;
    return this;
  }

  @Override
  public double getError() {
    return getGradientDescentTrainer().getError();
  }

  @Override
  public DAGNetwork getNet() {
    return getGradientDescentTrainer().getNet();
  }

}
