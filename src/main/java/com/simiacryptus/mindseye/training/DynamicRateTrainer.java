package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.DeltaBuffer;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.math.MultivariateOptimizer;
import com.simiacryptus.mindseye.util.Util;

public class DynamicRateTrainer {

  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);

  private double baseRate = .1;
  int currentIteration = 0;
  private double decayTolerance = 1e-3;
  int generationsSinceImprovement = 0;
  private final ChampionTrainer inner;
  int lastCalibratedIteration = Integer.MIN_VALUE;
  final int maxIterations = 1000;
  private double maxRate = 100;
  double minRate = 0;
  double monteCarloDecayStep = 0.9;
  double monteCarloMin = 0.5;
  private double mutationFactor = 1.;
  double rate = 0.5;
  double[] rates = null;
  private int recalibrationInterval = 10;
  int recalibrationThreshold = 0;
  private double stopError = 0;
  private boolean verbose = false;

  public DynamicRateTrainer() {
    this(new ChampionTrainer());
  }

  public DynamicRateTrainer(final ChampionTrainer inner) {
    this.inner = inner;
  }

  public MultivariateFunction asMetaF(final DeltaBuffer lessonVector, final double fraction) {
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[lessonVector.map.size()];

      @Override
      public double value(final double x[]) {
        final GradientDescentTrainer current = DynamicRateTrainer.this.getInner().getCurrent();
        final List<DeltaFlushBuffer> writeVectors = current.getCurrentNetworks().stream()
            .flatMap(n -> n.getNet().layers.stream())
            .map(n -> lessonVector.map.get(n))
            .filter(n -> null != n)
            .distinct()
            .sorted(Comparator.comparing(y -> y.getId()))
            .collect(Collectors.toList());
        final int layerCount = writeVectors.size();
        final double[] layerRates = Arrays.copyOf(x, layerCount);
        // double[] netRates = Arrays.copyOfRange(x, layerCount, current.getCurrentNetworks().size());
        if (current.getCurrentNetworks().size() > 1) {
          DynamicRateTrainer.log.debug("TODO: Optimize the rates of each network. Needs seperate delta buffers for each network within same layer obj!");
        }
        assert layerRates.length == this.pos.length;
        for (int i = 0; i < layerRates.length; i++) {
          final double prev = this.pos[i];
          final double next = layerRates[i];
          final double adj = next - prev;
          writeVectors.get(i).write(adj);
        }
        for (int i = 0; i < layerRates.length; i++) {
          this.pos[i] = layerRates[i];
        }
        final double[] calcError = current
            .calcError(current.evalTrainingData());
        final double err = Util.geomMean(calcError);
        if (isVerbose()) {
          DynamicRateTrainer.log.debug(String.format("f[%s] = %s (%s)", Arrays.toString(layerRates), err, Arrays.toString(calcError)));
        }
        return err;
      }
    };
    return f;
  }

  protected boolean calibrate() {
    final double last = error();
    boolean inBounds = false;
    PointValuePair optimum;
    final List<SupervisedTrainingParameters> nets = getInner().getCurrent().getCurrentNetworks();
    try {
      optimum = optimizeRates();
      nets.stream()
      .flatMap(n -> n.getNet().layers.stream())
      .distinct()
      .forEach(layer -> layer.setStatus(optimum.getValue()));
      this.rates = DoubleStream.of(optimum.getKey()).map(x -> x * this.rate).toArray();
      inBounds = DoubleStream.of(this.rates).allMatch(r -> getMaxRate() > r)
          && DoubleStream.of(this.rates).anyMatch(r -> this.minRate < r);
      if (inBounds)
      {
        this.lastCalibratedIteration = this.currentIteration;
        final double err = trainOnce();
        final double improvement = last - err;
        if (isVerbose()) {
          DynamicRateTrainer.log.debug(String.format("Adjusting rates by %s: (%s->%s - %s improvement)", Arrays.toString(this.rates), last, err, improvement));
        }
        return improvement > 0;
      }
    } catch (final Exception e) {
      if (isVerbose()) {
        DynamicRateTrainer.log.debug("Error calibrating", e);
      }
    }
    if (isVerbose()) {
      DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", Arrays.toString(this.rates),
          Arrays.toString(getInner().getCurrent().getError())));
    }
    return false;
  }

  public double error() {
    return getInner().getCurrent().error();
  }

  public double getBaseRate() {
    return this.baseRate;
  }

  public GradientDescentTrainer getBest() {
    return getInner().getBest();
  }

  public double getDecayTolerance() {
    return this.decayTolerance;
  }

  public int getGenerationsSinceImprovement() {
    return this.generationsSinceImprovement;
  }

  public ChampionTrainer getInner() {
    return this.inner;
  }

  public List<NNLayer> getLayers() {
    return getInner().getLayers();
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

  public List<PipelineNetwork> getNetwork() {
    return this.inner.getNetwork();
  }

  public double getRate() {
    return this.rate;
  }

  public int getRecalibrationThreshold() {
    return this.recalibrationThreshold;
  }

  public double getStopError() {
    return this.stopError;
  }

  public boolean isVerbose() {
    // return true;
    return this.verbose;
  }
  
  public synchronized PointValuePair optimizeRates() {
    final double[] prev = getInner().getCurrent().calcError(getInner().getCurrent().evalTrainingData());
    assert 0 < getInner().getCurrent().getCurrentNetworks().size();
    final DeltaBuffer lessonVector = getInner().getCurrent().learn(getInner().getCurrent().evalTrainingData(), new DeltaBuffer());
    // final double[] one = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    double fraction = 1.;
    PointValuePair x = null;
    do {
      final MultivariateFunction f = asMetaF(lessonVector, fraction);
      x = new MultivariateOptimizer(f).setMaxRate(getMaxRate()).minimize(lessonVector.map.size()); // May or may not be cloned before evaluations
      f.value(x.getFirst()); // Leave in optimal state
      fraction *= this.monteCarloDecayStep;
    } while (fraction > this.monteCarloMin && new ArrayRealVector(x.getFirst()).getL1Norm() == 0);
    // f.value(new double[dims]); // Reset to original state
    final double[] calcError = getInner().getCurrent().calcError(getInner().getCurrent().evalTrainingData());
    getInner().getCurrent().setError(calcError);
    if (this.verbose) {
      DynamicRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s", Arrays.toString(x.getKey()), x.getValue(),
          Arrays.toString(prev), Arrays.toString(calcError)));
    }
    return x;
  }

  public DynamicRateTrainer setBaseRate(final double baseRate) {
    this.baseRate = baseRate;
    return this;
  }

  public boolean setDecayTolerance(final double decayTolerance) {
    this.decayTolerance = decayTolerance;
    return true;
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

  public DynamicRateTrainer setStopError(final double stopError) {
    this.stopError = stopError;
    return this;
  }

  public DynamicRateTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    getInner().setVerbose(verbose);
    return this;
  }

  public double trainOnce() {
    getInner().step(this.rates);
    getInner().updateBest();
    final double error = error();
    final List<SupervisedTrainingParameters> nets = getInner().getCurrent().getCurrentNetworks();
    nets.stream()
    .flatMap(n -> n.getNet().layers.stream())
    .distinct()
    .forEach(layer -> layer.setStatus(error));
    return error;
  }

  public boolean trainToLocalOptimum() {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (true) {
      if (getStopError() > error()) {
        DynamicRateTrainer.log.debug("Target error reached: " + error());
        return false;
      }
      if (this.maxIterations <= this.currentIteration++) {
        DynamicRateTrainer.log.debug("Maximum steps reached");
        return false;
      }
      if (this.lastCalibratedIteration < this.currentIteration - this.recalibrationInterval) {
        if (isVerbose()) {
          DynamicRateTrainer.log.debug("Recalibrating learning rate due to interation schedule at " + this.currentIteration);
        }
        // calibrate();
        if (!calibrate()) {
          DynamicRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
          return false;
        }
      }
      final double best = getInner().getBest().error();
      final double last = getInner().getCurrent().error();
      final double next = trainOnce();
      if (next / last < 1 + getDecayTolerance() || next / best < 1 + 5 * getDecayTolerance())
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
          if (!calibrate()) {
            DynamicRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
            return false;
          }
          this.generationsSinceImprovement = 0;
        }
      }
    }
  }

}
