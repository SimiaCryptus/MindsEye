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
  int generationsSinceImprovement = 0;
  
  private final ChampionTrainer inner;
  int lastCalibratedIteration = Integer.MIN_VALUE;
  final int maxIterations = 1000;
  private double maxRate = 100;
  double minRate = 0;
  private double mutationFactor = 1.;
  double rate = 0.5;
  private int recalibrationInterval = 10;
  int recalibrationThreshold = 0;
  
  private boolean verbose = false;
  
  private double stopError = 0;
  
  public DynamicRateTrainer() {
    this(new ChampionTrainer());
  }
  
  public DynamicRateTrainer(final ChampionTrainer inner) {
    this.inner = inner;
  }
  
  double[] rates = null;
  protected boolean calibrate() {
    final double last = error();
    boolean inBounds = false;
    PointValuePair optimum;
    List<SupervisedTrainingParameters> nets = this.getInner().getCurrent().getCurrentNetworks();
    try {
      optimum = optimizeRates();
      nets.stream()
          .flatMap(n -> n.getNet().layers.stream())
          .distinct()
          .forEach(layer -> layer.setStatus((double) optimum.getValue()));
      rates = DoubleStream.of(optimum.getKey()).map(x -> x * this.rate).toArray();
      inBounds = DoubleStream.of(rates).allMatch(r -> this.getMaxRate() > r)
          && DoubleStream.of(rates).anyMatch(r -> this.minRate < r);
      if (inBounds)
      {
        this.lastCalibratedIteration = this.currentIteration;
        final double err = trainOnce();
        final double improvement = last - err;
        if (isVerbose()) {
          DynamicRateTrainer.log.debug(String.format("Adjusting rates by %s: (%s->%s - %s improvement)", Arrays.toString(rates), last, err, improvement));
        }
        return improvement > 0;
      }
    } catch (final Exception e) {
      if (isVerbose()) {
        DynamicRateTrainer.log.debug("Error calibrating", e);
      }
    }
    if (isVerbose()) {
      DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", Arrays.toString(rates),
          Arrays.toString(this.getInner().getCurrent().getError())));
    }
    return false;
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
  
  double monteCarloMin = 0.5;
  double monteCarloDecayStep = 0.9;
  
  public synchronized PointValuePair optimizeRates() {
    final double[] prev = this.getInner().getCurrent().calcError(this.getInner().getCurrent().evalTrainingData());
    assert 0 < this.getInner().getCurrent().getCurrentNetworks().size();
    DeltaBuffer lessonVector = this.getInner().getCurrent().learn(this.getInner().getCurrent().evalTrainingData(), new DeltaBuffer());
    // final double[] one = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    double fraction = 1.;
    PointValuePair x = null;
    do {
      final MultivariateFunction f = asMetaF(lessonVector, fraction);
      x = new MultivariateOptimizer(f).setMaxRate(getMaxRate()).minimize(lessonVector.map.size()); // May or may not be cloned before evaluations
      f.value(x.getFirst()); // Leave in optimal state
      fraction *= monteCarloDecayStep;
    } while (fraction > monteCarloMin && new ArrayRealVector(x.getFirst()).getL1Norm() == 0);
    // f.value(new double[dims]); // Reset to original state
    final double[] calcError = this.getInner().getCurrent().calcError(this.getInner().getCurrent().evalTrainingData());
    this.getInner().getCurrent().setError(calcError);
    if (this.verbose) {
      DynamicRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s", Arrays.toString(x.getKey()), x.getValue(),
          Arrays.toString(prev), Arrays.toString(calcError)));
    }
    return x;
  }
  
  public MultivariateFunction asMetaF(DeltaBuffer lessonVector, double fraction) {
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[lessonVector.map.size()];
      
      @Override
      public double value(final double x[]) {
        GradientDescentTrainer current = DynamicRateTrainer.this.getInner().getCurrent();
        List<DeltaFlushBuffer> writeVectors = current.getCurrentNetworks().stream()
            .flatMap(n -> n.getNet().layers.stream())
            .map(n -> lessonVector.map.get(n))
            .filter(n -> null != n)
            .distinct()
            .sorted(Comparator.comparing(y->y.getId()))
            .collect(Collectors.toList());
        int layerCount = writeVectors.size();
        double[] layerRates = Arrays.copyOf(x, layerCount);
        // double[] netRates = Arrays.copyOfRange(x, layerCount, current.getCurrentNetworks().size());
        if (current.getCurrentNetworks().size() > 1)
          log.debug("TODO: Optimize the rates of each network. Needs seperate delta buffers for each network within same layer obj!");
        assert layerRates.length == this.pos.length;
        for (int i = 0; i < layerRates.length; i++) {
          double prev = this.pos[i];
          double next = layerRates[i];
          double adj = next - prev;
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
  
  public double trainOnce() {
    this.getInner().step(rates);
    this.getInner().updateBest();
    double error = error();
    List<SupervisedTrainingParameters> nets = this.getInner().getCurrent().getCurrentNetworks();
    nets.stream()
        .flatMap(n -> n.getNet().layers.stream())
        .distinct()
        .forEach(layer -> layer.setStatus(error));
    return error;
  }
  
  private double decayTolerance = 1e-3;
  
  public boolean trainToLocalOptimum() {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (true) {
      if (this.getStopError() > error()) {
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
      final double best = this.getInner().getBest().error();
      final double last = this.getInner().getCurrent().error();
      double next = trainOnce();
      if ((next / last) < 1 + getDecayTolerance() || (next / best) < 1 + 5 * getDecayTolerance())
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
  
  public double getDecayTolerance() {
    return decayTolerance;
  }
  
  public boolean setDecayTolerance(double decayTolerance) {
    this.decayTolerance = decayTolerance;
    return true;
  }
  
}
