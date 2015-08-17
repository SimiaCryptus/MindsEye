package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.math.MultivariateOptimizer;

public class DynamicRateTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);

  private double baseRate = .1;
  int currentIteration = 0;
  int generationsSinceImprovement = 0;

  private final ChampionTrainer inner;
  int lastCalibratedIteration = Integer.MIN_VALUE;
  final int maxIterations = 10000;
  double maxRate = 5e4;
  double minRate = 0;
  private double mutationFactor = 1.;
  double rate = 1;
  private int recalibrationInterval = 15;
  int recalibrationThreshold = 0;
  
  private boolean verbose = false;

  private double stopError = 0;
  
  public DynamicRateTrainer() {
    this(new ChampionTrainer());
  }
  
  public DynamicRateTrainer(final ChampionTrainer inner) {
    this.inner = inner;
  }
  
  protected boolean calibrate() {
    final double last = error();
    List<DeltaTransaction> deltaObjs = null;
    double[] adjustment = null;
    boolean inBounds = false;
    try {
      final GradientDescentTrainer current = this.getInner().getCurrent();
      deltaObjs = current.getCurrentNetworks().stream()
          .flatMap(n -> n.getNet().layers.stream())
          .filter(l -> l instanceof DeltaTransaction)
          .map(l -> (DeltaTransaction) l)
          .filter(x -> !x.isFrozen())
          .distinct().collect(Collectors.toList());
      for (int i = 0; i < deltaObjs.size(); i++) {
        deltaObjs.get(i).setRate(getBaseRate());
      }
      final double[] localMin = trainLineSearch(deltaObjs.size());
      adjustment = DoubleStream.of(localMin).map(x -> x * this.rate).toArray();
      inBounds = DoubleStream.of(adjustment).allMatch(r -> this.maxRate > r)
          && DoubleStream.of(adjustment).anyMatch(r -> this.minRate < r);
    } catch (final Exception e) {
      if (isVerbose()) {
        DynamicRateTrainer.log.debug("Error calibrating", e);
      }
    }
    if (inBounds)
    {
      for (int i = 0; i < deltaObjs.size(); i++) {
        final DeltaTransaction deltaTransaction = deltaObjs.get(i);
        deltaTransaction.setRate(adjustment[i] * deltaTransaction.getRate());
      }
      this.lastCalibratedIteration = this.currentIteration;
      final double err = trainOnce();
      final double improvement = last - err;
      if (isVerbose()) {
        DynamicRateTrainer.log.debug(String.format("Adjusting rates by %s: (%s->%s - %s improvement)", Arrays.toString(adjustment), last, err, improvement));
      }
      return improvement > 0;
    } else {
      if (isVerbose()) {
        DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", Arrays.toString(adjustment),
            Arrays.toString(this.getInner().getCurrent().getError())));
      }
      return false;
    }
  }

  public double error() {
    return this.getInner().getCurrent().error();
  }

  public double getBaseRate() {
    return this.baseRate;
  }
  
  public GradientDescentTrainer getBest() {
    return this.getInner().getBest();
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
  
  public double[] getRates() {
    return this.getInner().getCurrent().getRates();
  }
  
  public int getRecalibrationThreshold() {
    return this.recalibrationThreshold;
  }
  
  public boolean isVerbose() {
    // return true;
    return this.verbose;
  }
  
  public DynamicRateTrainer setBaseRate(final double baseRate) {
    this.baseRate = baseRate;
    return this;
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
    this.getInner().setVerbose(verbose);
    return this;
  }
  
  public synchronized double[] trainLineSearch(final int dims) {
    assert 0 < this.getInner().getCurrent().getCurrentNetworks().size();
    final double[] prev = this.getInner().getCurrent().getError();
    this.getInner().getCurrent().learn(this.getInner().getCurrent().evalTrainingData());
    // final double[] one = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[dims];

      @Override
      public double value(final double x[]) {
        final double[] diff = new double[x.length];
        for (int i = 0; i < diff.length; i++) {
          diff[i] = x[i] - this.pos[i];
        }
        final List<DeltaTransaction> deltaObjs = DynamicRateTrainer.this.getInner().getCurrent().getCurrentNetworks().stream()
            .flatMap(n -> n.getNet().layers.stream())
            .filter(l -> l instanceof DeltaTransaction)
            .map(l -> (DeltaTransaction) l)
            .filter(l -> !l.isFrozen())
            .distinct().collect(Collectors.toList());
        assert diff.length == deltaObjs.size();
        assert diff.length == this.pos.length;
        for (int i = 0; i < diff.length; i++) {
          deltaObjs.get(i).write(diff[i]);
        }
        for (int i = 0; i < diff.length; i++) {
          this.pos[i] += diff[i];
        }
        final double[] calcError = DynamicRateTrainer.this.getInner().getCurrent().calcError(DynamicRateTrainer.this.getInner().getCurrent().evalTrainingData());
        final double err = Util.geomMean(calcError);
        if (isVerbose()) {
          DynamicRateTrainer.log.debug(String.format("f[%s] = %s (%s)", Arrays.toString(x), err, Arrays.toString(calcError)));
        }
        return err;
      }
    };
    final PointValuePair x = new MultivariateOptimizer(f).minimize(dims); // May or may not be cloned before evaluations
    f.value(x.getFirst()); // Reset to original state
    // f.value(new double[dims]); // Reset to original state
    final double[] calcError = this.getInner().getCurrent().calcError(this.getInner().getCurrent().evalTrainingData());
    this.getInner().getCurrent().setError(calcError);
    if (this.verbose) {
      DynamicRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s", Arrays.toString(x.getKey()), x.getValue(),
          Arrays.toString(prev), Arrays.toString(calcError)));
    }
    return x.getKey();
  }
  
  public double trainOnce() {
    this.getInner().step();
    this.getInner().updateBest();
    return error();
  }
  
  public boolean trainToLocalOptimum() {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (this.maxIterations > this.currentIteration++ && this.getStopError() < error()) {
      if (this.lastCalibratedIteration < this.currentIteration - this.recalibrationInterval) {
        if (isVerbose()) {
          DynamicRateTrainer.log.debug("Recalibrating learning rate due to interation schedule at " + this.currentIteration);
        }
        calibrate();
        // if (!calibrate()) return false;
      }
      final double last = error();
      final double improvement = last - trainOnce();
      if (improvement > 0)
      {
        this.generationsSinceImprovement = 0;
      }
      else
      {
        if (this.recalibrationThreshold < this.generationsSinceImprovement++)
        {
          if (isVerbose()) {
            DynamicRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
          }
          if (!calibrate()) return false;
          this.generationsSinceImprovement = 0;
        }
      }
    }
    DynamicRateTrainer.log.debug("Maximum steps reached");
    return false;
  }

  public ChampionTrainer getInner() {
    return inner;
  }

  public double getStopError() {
    return stopError;
  }

  public DynamicRateTrainer setStopError(double stopError) {
    this.stopError = stopError;
    return this;
  }

  public List<NNLayer> getLayers() {
    return getInner().getLayers();
  }

  public List<PipelineNetwork> getNetwork() {
    return inner.getNetwork();
  }
  
}
