package com.simiacryptus.mindseye.training;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MacroTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(MacroTrainer.class);

  private int currentGeneration = 0;
  private final MutationTrainer inner;
  private int maxIterations = 1000;
  private double stopError = 0.1;
  private boolean verbose = false;
  
  public MacroTrainer() {
    this(new MutationTrainer());
  }
  
  public MacroTrainer(final MutationTrainer inner) {
    this.inner = inner;
  }

  public boolean continueTraining() {
    if (this.maxIterations < this.currentGeneration) {
      if (this.verbose) {
        MacroTrainer.log.debug("Reached max iterations: " + this.currentGeneration);
      }
      return false;
    }
    if (this.inner.error() < this.stopError) {
      if (this.verbose) {
        MacroTrainer.log.debug("Reached convergence: " + this.inner.error());
      }
      return false;
    }
    return true;
  }
  
  public GradientDescentTrainer getBest() {
    return this.inner.getBest();
  }
  
  public DynamicRateTrainer getInner() {
    return this.inner.inner;
  }
  
  public int getMaxIterations() {
    return this.maxIterations;
  }
  
  public double getStopError() {
    return this.stopError;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public void mutate(final double mutationAmount) {
    if (this.verbose) {
      MacroTrainer.log.debug(String.format("Mutating %s by %s", this.inner, mutationAmount));
    }
    this.inner.inner.inner.current.mutate(mutationAmount);
  }
  
  public MacroTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
  public MacroTrainer setMutationAmount(final double mutationAmount) {
    this.inner.setMutationFactor(mutationAmount);
    return this;
  }
  
  public MacroTrainer setStopError(final double stopError) {
    this.stopError = stopError;
    return this;
  }
  
  public MacroTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    this.inner.setVerbose(verbose);
    return this;
  }
  
  public Double train() {
    final long startMs = System.currentTimeMillis();
    this.currentGeneration = 0;
    this.inner.inner.inner.current.mutate(1);
    while (continueTraining())
    {
      this.currentGeneration++;
      this.inner.train(stopError);
      if (this.verbose)
      {
        MacroTrainer.log.debug(String.format("Trained Iteration %s Error: %s (%s) with rate %s",
            this.currentGeneration, this.inner.error(), Arrays.toString(this.inner.inner.inner.current.error), this.inner.inner.inner.current.getRate()));
      }
    }
    MacroTrainer.log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", this.inner.error(),
        (System.currentTimeMillis() - startMs) / 1000.,
        this.currentGeneration));
    GradientDescentTrainer best = this.inner.inner.inner.best;
    return null==best?Double.POSITIVE_INFINITY:best.error();
  }
  
}
