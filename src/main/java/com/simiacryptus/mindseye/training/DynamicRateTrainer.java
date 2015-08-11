package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.learning.DeltaTransaction;

public class DynamicRateTrainer {

  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);
  
  int currentIteration = 0;
  int generationsSinceImprovement = 0;
  int lastCalibratedIteration = Integer.MIN_VALUE;
  
  public final ChampionTrainer inner;
  double maxRate = 5e4;
  double minRate = 1e-10;
  private double mutationFactor = 1.;
  double rate = 1/Math.E;
  private int recalibrationInterval = 15;
  int recalibrationThreshold = 3;
  private boolean verbose = false;

  public DynamicRateTrainer() {
    this(new ChampionTrainer());
  }

  public DynamicRateTrainer(final ChampionTrainer inner) {
    this.inner = inner;
  }

  protected boolean calibrate() {
    List<DeltaTransaction> deltaObjs = null;
    double[] adjustment = null;
    double[] newRate = null;
    boolean inBounds = false;
    try {
      deltaObjs = this.inner.current.currentNetworks.stream()
          .flatMap(n -> n.getNet().layers.stream())
          .filter(l -> l instanceof DeltaTransaction)
          .map(l -> (DeltaTransaction) l)
          .filter(x->!x.isFrozen())
          .distinct().collect(Collectors.toList());
      final double[] localMin = this.inner.current.copy().clearMomentum().trainLineSearch(deltaObjs.size());
      adjustment = DoubleStream.of(localMin).map(x -> x * this.rate).toArray();
      newRate = DoubleStream.of(adjustment).map(x -> x * this.inner.current.getRate()).toArray();
      inBounds = DoubleStream.of(newRate).anyMatch(r -> this.maxRate > r && this.minRate < r);
    } catch (final Exception e) {
      if (this.verbose) {
        DynamicRateTrainer.log.debug("Error calibrating", e);
      }
    }
    if (inBounds)
    {
      if (this.verbose) {
        DynamicRateTrainer.log.debug(String.format("Adjusting rate by %s: %s", adjustment, newRate));
      }
      for (int i = 0; i < deltaObjs.size(); i++) {
        deltaObjs.get(i).setRate(newRate[i]);
      }
      this.inner.train();
      this.inner.updateBest();
      this.lastCalibratedIteration = this.currentIteration;
      return true;
    } else {
      return false;
    }
  }
  
  public double error() {
    return this.inner.current.error();
  }

  public GradientDescentTrainer getBest() {
    return this.inner.getBest();
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

  public double getMutationFactor() {
    return this.mutationFactor;
  }

  public double getRate() {
    return this.rate;
  }

  public int getRecalibrationThreshold() {
    return this.recalibrationThreshold;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public DynamicRateTrainer setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    this.generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }

  public DynamicRateTrainer setMaxRate(final double maxRate) {
    this.maxRate = maxRate;
    return this;
  }

  public DynamicRateTrainer setMinRate(final double minRate) {
    this.minRate = minRate;
    return this;
  }
  
  public void setMutationFactor(final double mutationRate) {
    this.mutationFactor = mutationRate;
  }
  
  public DynamicRateTrainer setRate(final double rate) {
    this.rate = rate;
    return this;
  }
  
  public DynamicRateTrainer setRecalibrationThreshold(final int recalibrationThreshold) {
    this.recalibrationThreshold = recalibrationThreshold;
    return this;
  }
  
  public DynamicRateTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    this.inner.setVerbose(verbose);
    return this;
  }
  
  public boolean train() {
    currentIteration = 0;
    generationsSinceImprovement = 0;
    lastCalibratedIteration = Integer.MIN_VALUE;
    if (this.lastCalibratedIteration < this.currentIteration++ - this.recalibrationInterval) {
      if (this.verbose) {
        DynamicRateTrainer.log.debug("Recalibrating learning rate due to interation schedule");
      }
      if(!calibrate()) return false;
    }
    final double lastError = this.inner.current.error();
    this.inner.train();
    final double resultError = this.inner.current.error();
    if (Double.isFinite(lastError) && Double.isFinite(resultError)) {
      if (resultError >= lastError)
      {
        if (this.recalibrationThreshold < this.generationsSinceImprovement++)
        {
          if (this.verbose) {
            DynamicRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
          }
          calibrate();
          this.generationsSinceImprovement = 0;
        }
      }
      else
      {
        this.generationsSinceImprovement = 0;
      }
    }
    // updateRate(lastError, resultError);
    return true;
  }

}
