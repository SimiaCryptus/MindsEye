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
  double minRate = 0;
  private double mutationFactor = 1.;
  double rate = 1;
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
    double last = this.inner.current.error();
    List<DeltaTransaction> deltaObjs = null;
    double[] adjustment = null;
    double[] newRate = null;
    boolean inBounds = false;
    try {
      GradientDescentTrainer current = this.inner.current;
      deltaObjs = current.currentNetworks.stream()
          .flatMap(n -> n.getNet().layers.stream())
          .filter(l -> l instanceof DeltaTransaction)
          .map(l -> (DeltaTransaction) l)
          .filter(x->!x.isFrozen())
          .distinct().collect(Collectors.toList());
      for (int i = 0; i < deltaObjs.size(); i++) {
        deltaObjs.get(i).setRate(1);
      }
      GradientDescentTrainer clone = current.copy().clearMomentum();
      final double[] localMin = clone.trainLineSearch(deltaObjs.size());
      adjustment = DoubleStream.of(localMin).map(x -> x * this.rate).toArray();
      newRate = DoubleStream.of(adjustment).map(x -> x * current.getRate()).toArray();
      inBounds = DoubleStream.of(newRate).allMatch(r -> this.maxRate > r) 
          && DoubleStream.of(newRate).anyMatch(r -> this.minRate < r);
    } catch (final Exception e) {
      if (this.isVerbose()) {
        DynamicRateTrainer.log.debug("Error calibrating", e);
      }
    }
    if (inBounds)
    {
      for (int i = 0; i < deltaObjs.size(); i++) {
        deltaObjs.get(i).setRate(newRate[i]);
      }
      this.lastCalibratedIteration = this.currentIteration;
      double improvement = last - trainOnce();
      if (this.isVerbose()) {
        DynamicRateTrainer.log.debug(String.format("Adjusting rates by %s: %s (%s - %s improvement)", Arrays.toString(adjustment), Arrays.toString(newRate), error(), improvement));
      }
      return improvement>0;
    } else {
      if (this.isVerbose()) {
        DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", Arrays.toString(newRate), Arrays.toString(this.inner.current.getError())));
      }
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
    //return true;
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
  
  final int maxIterations=10000;
  public boolean trainToLocalOptimum() {
    currentIteration = 0;
    generationsSinceImprovement = 0;
    lastCalibratedIteration = Integer.MIN_VALUE;
    while (maxIterations>currentIteration++) {
      if (this.lastCalibratedIteration < this.currentIteration - this.recalibrationInterval) {
        if (this.isVerbose()) {
          DynamicRateTrainer.log.debug("Recalibrating learning rate due to interation schedule at " + this.currentIteration);
        }
        calibrate();
        //if (!calibrate()) return false;
      }
      double last = this.inner.current.error();
      double improvement = last - trainOnce();
      if (improvement > 0)
      {
        this.generationsSinceImprovement = 0;
      }
      else
      {
        if (this.recalibrationThreshold < this.generationsSinceImprovement++)
        {
          if (this.isVerbose()) {
            DynamicRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
          }
          if (!calibrate()) return false;
          this.generationsSinceImprovement = 0;
        }
      }
    }
    log.debug("Maximum steps reached");
    return false;
  }

  public double trainOnce() {
    this.inner.step();
    this.inner.updateBest();
    return this.inner.current.error();
  }

}
