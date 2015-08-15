package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;

public class MutationTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(MutationTrainer.class);

  private int currentGeneration = 0;
  public final DynamicRateTrainer inner;
  private int maxIterations = 1000;
  private double mutationFactor = 1.;
  private double stopError = 0.1;
  private boolean verbose = false;
  
  public MutationTrainer() {
    this(new ChampionTrainer());
  }
  
  public MutationTrainer(final ChampionTrainer inner) {
    this.inner = new DynamicRateTrainer(inner);
  }
  
  public boolean continueTraining() {
    if (this.maxIterations < this.currentGeneration) {
      if (this.verbose) {
        MutationTrainer.log.debug("Reached max iterations: " + this.currentGeneration);
      }
      return false;
    }
    if (this.inner.error() < this.stopError) {
      if (this.verbose) {
        MutationTrainer.log.debug("Reached convergence: " + this.inner.error());
      }
      return false;
    }
    return true;
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
  
  public DynamicRateTrainer getInner() {
    return this.inner;
  }
  
  public int getMaxIterations() {
    return this.maxIterations;
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
    return this.inner.getRate();
  }

  public int getRecalibrationThreshold() {
    return this.inner.recalibrationThreshold;
  }

  public double getStopError() {
    return this.stopError;
  }

  public boolean isVerbose() {
    return this.verbose;
  }
  
  public BiasLayer mutate(final BiasLayer l, final double amount) {
    final double[] a = l.bias;
    Random random = Util.R.get();
    for (int i = 0; i < a.length; i++)
    {
      if (random.nextDouble() < amount) {
        //a[i] = mutationAmplitude * random.nextGaussian();
      }
    }
    return l;
  }
  
  public DenseSynapseLayer mutate(final DenseSynapseLayer l, final double amount) {
    final double[] a = l.weights.getData();
    Random random = Util.R.get();
    for (int i = 0; i < a.length; i++)
    {
      if (random.nextDouble() < amount) {
        a[i] = mutationAmplitude * random.nextGaussian();
      }
    }
    return l;
  }
  
  public void mutate(final double amount) {
    if (this.verbose) {
      MutationTrainer.log.debug(String.format("Mutating %s by %s", this.inner, amount));
    }
    List<NNLayer> layers = this.inner.inner.current.currentNetworks.stream().flatMap(x->x.getNet().layers.stream()).distinct().collect(Collectors.toList());
    layers.stream()
        .filter(l -> (l instanceof DenseSynapseLayer))
        .map(l -> (DenseSynapseLayer) l)
        .filter(l -> !l.isFrozen())
        .forEach(l -> mutate(l, amount));
    layers.stream()
        .filter(l -> (l instanceof BiasLayer))
        .map(l -> (BiasLayer) l)
        .filter(l -> !l.isFrozen())
        .forEach(l -> mutate(l, amount));
    this.inner.inner.current.setError(null);
  }
  
  double mutationAmplitude = 2.;

  public void mutateBest() {
    this.inner.generationsSinceImprovement = this.inner.recalibrationThreshold - 1;
    this.inner.inner.revert();
    mutate(getMutationFactor());
    this.inner.lastCalibratedIteration = this.inner.currentIteration;// - (this.recalibrationInterval + 2);
  }
  
  public MutationTrainer setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    this.inner.generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }

  public MutationTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
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

  public MutationTrainer setMutationAmount(final double mutationAmount) {
    this.inner.setMutationFactor(mutationAmount);
    return this;
  }

  public void setMutationFactor(final double mutationRate) {
    this.mutationFactor = mutationRate;
  }

  public MutationTrainer setRate(final double rate) {
    this.inner.setRate(rate);
    return this;
  }
  
  public MutationTrainer setRecalibrationThreshold(final int recalibrationThreshold) {
    this.inner.recalibrationThreshold = recalibrationThreshold;
    return this;
  }
  
  public MutationTrainer setStopError(final double stopError) {
    this.stopError = stopError;
    return this;
  }
  
  public MutationTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    this.inner.setVerbose(verbose);
    return this;
  }
  
  public Double train() {
    final long startMs = System.currentTimeMillis();
    this.currentGeneration = 0;
    while (continueTraining())
    {
      if (0 == this.currentGeneration++) {
        mutate(1);
      } else {
        mutateBest();
      }
      this.inner.trainToLocalOptimum();
      if (this.verbose)
      {
        MutationTrainer.log.debug(String.format("Trained Iteration %s Error: %s (%s) with rate %s",
            this.currentGeneration, this.inner.error(), Arrays.toString(this.inner.inner.current.getError()), this.inner.inner.current.getRate()));
      }
    }
    MutationTrainer.log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", this.inner.error(),
        (System.currentTimeMillis() - startMs) / 1000.,
        this.currentGeneration));
    final GradientDescentTrainer best = this.inner.inner.best;
    return null == best ? Double.POSITIVE_INFINITY : best.error();
  }

  public double getMutationAmplitude() {
    return this.mutationAmplitude;
  }

  public MutationTrainer setMutationAmplitude(final double mutationAmplitude) {
    this.mutationAmplitude = mutationAmplitude;
    return this;
  }
}
