package com.simiacryptus.mindseye.training;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MacroTrainer {

  private static final Logger log = LoggerFactory.getLogger(MacroTrainer.class);
  
  private int currentGeneration = 0;
  private final DynamicRateTrainer inner;
  private int maxIterations = 1000;
  private double stopError = 0.1;
  private boolean verbose = false;

  public MacroTrainer() {
    this(new DynamicRateTrainer());
  }

  public MacroTrainer(DynamicRateTrainer inner) {
    this.inner = inner;
  }
  
  public boolean continueTraining() {
    if(maxIterations < this.currentGeneration) {
      if(verbose) log.debug("Reached max iterations: " + this.currentGeneration);
      return false;
    }
    if(inner.error() < stopError) {
      if(verbose) log.debug("Reached convergence: " + inner.error());
      return false;
    }
    return true;
  }

  public GradientDescentTrainer getBest() {
    return this.inner.getBest();
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public double getStopError() {
    return stopError;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public void mutate(final double mutationAmount) {
    if (this.verbose) {
      MacroTrainer.log.debug(String.format("Mutating %s by %s", this.inner, mutationAmount));
    }
    this.inner.inner.current.mutate(mutationAmount);
  }

  public MacroTrainer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }

  public MacroTrainer setMutationAmount(final double mutationAmount) {
    inner.setMutationFactor(mutationAmount);
    return this;
  }

  public MacroTrainer setStopError(double stopError) {
    this.stopError = stopError;
    return this;
  }

  public MacroTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    inner.setVerbose(true);
    return this;
  }

  public Double train() {
    final long startMs = System.currentTimeMillis();
    this.currentGeneration = 0;
    inner.inner.current.mutate(1);
    while (continueTraining())
    {
      this.currentGeneration++;
      inner.train();
      if (this.verbose)
      {
        MacroTrainer.log.debug(String.format("Trained Iteration %s Error: %s (%s) with rate %s",
            this.currentGeneration, inner.error(), Arrays.toString(inner.inner.current.error), inner.inner.current.getRate()));
      }
    }
    MacroTrainer.log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", inner.error(), (System.currentTimeMillis() - startMs) / 1000.,
        this.currentGeneration));
    return this.inner.inner.best.error();
  }

  public DynamicRateTrainer getInner() {
    return inner;
  }

}
