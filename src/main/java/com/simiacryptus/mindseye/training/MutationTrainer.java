package com.simiacryptus.mindseye.training;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MutationTrainer {

  private static final Logger log = LoggerFactory.getLogger(MutationTrainer.class);
  
  public final DynamicRateTrainer inner;
  private double mutationFactor = 1.;
  private boolean verbose = true;

  public MutationTrainer() {
    this(new ChampionTrainer());
  }

  public MutationTrainer(final ChampionTrainer inner) {
    this.inner = new DynamicRateTrainer(inner);
  }
  
  public double error() {
    return this.inner.inner.current.error();
  }

  public GradientDescentTrainer getBest() {
    return this.inner.getBest();
  }

  public int getGenerationsSinceImprovement() {
    return this.inner.generationsSinceImprovement;
  }

  public double getMaxRate() {
    return this.inner.maxRate;
  }

  public double getMinRate() {
    return this.inner.minRate;
  }

  public double getMutationFactor() {
    return this.mutationFactor;
  }

  public double getRate() {
    return this.inner.rate;
  }

  public int getRecalibrationThreshold() {
    return this.inner.recalibrationThreshold;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public MutationTrainer setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    this.inner.generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }

  public MutationTrainer setMaxRate(final double maxRate) {
    this.inner.maxRate = maxRate;
    return this;
  }

  public MutationTrainer setMinRate(final double minRate) {
    this.inner.minRate = minRate;
    return this;
  }
  
  public void setMutationFactor(final double mutationRate) {
    this.mutationFactor = mutationRate;
  }
  
  public MutationTrainer setRate(final double rate) {
    this.inner.rate = rate;
    return this;
  }
  
  public MutationTrainer setRecalibrationThreshold(final int recalibrationThreshold) {
    this.inner.recalibrationThreshold = recalibrationThreshold;
    return this;
  }
  
  public MutationTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    this.inner.setVerbose(verbose);
    return this;
  }
  
  public void train(double stopError) {
    this.inner.trainToLocalOptimum();
    while(null == this.inner.inner.best || this.inner.inner.best.error() > stopError) {
      if (this.verbose) {
        log.debug(String.format("Local Optimum reached - gradient not useful. Mutating."));
      }
      this.inner.generationsSinceImprovement = this.inner.recalibrationThreshold-1;
      this.inner.inner.revert();
      this.inner.inner.current.mutate(getMutationFactor());
      this.inner.lastCalibratedIteration = this.inner.currentIteration;// - (this.recalibrationInterval + 2);
      this.inner.trainToLocalOptimum();
    }
    
  }

}
