package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.DeltaFlushBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.MultivariateOptimizer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class DynamicRateTrainer {
  public static class RateMonitor {
    public final long startTime = System.currentTimeMillis();
    private long lastUpdateTime = System.currentTimeMillis();
    private final double halfLifeMs;
    private double counter0 = 0;
    private double counter1 = 0;
    
    public RateMonitor(double halfLifeMs) {
      super();
      this.halfLifeMs = halfLifeMs;
    }

    public double add(double value) {
      long prevUpdateTime = lastUpdateTime;
      long now = System.currentTimeMillis();
      lastUpdateTime = now;
      long elapsedMs = now - prevUpdateTime;
      double elapsedHalflifes = elapsedMs / halfLifeMs;
      counter0+=elapsedMs;
      counter1+=value;
      double v = counter1/counter0;
      double f = Math.pow(0.5, elapsedHalflifes);
      counter0*=f;
      counter1*=f;
      return v;
    }
  }

  public static class UniformAdaptiveRateParams {
    public final double endRate;
    public final double alpha;
    public final double beta;
    public final double startRate;
    public final double terminalLearningRate;
    public final double terminalETA;

    public UniformAdaptiveRateParams(double startRate, double endRate, double alpha, double beta, double convergence, double terminalETA) {
      this.endRate = endRate;
      this.alpha = alpha;
      this.beta = beta;
      this.startRate = startRate;
      this.terminalLearningRate = convergence;
      this.terminalETA = terminalETA;
    }
  }
  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);

  int currentIteration = 0;
  int generationsSinceImprovement = 0;
  private final GradientDescentTrainer inner = new GradientDescentTrainer();
  int lastCalibratedIteration = Integer.MIN_VALUE;
  int maxIterations = 100;
  private double maxRate = 10000;
  private double minRate = 0;
  private int recalibrationInterval = 10;
  private int recalibrationThreshold = 0;
  private double stopError = 0;
  private boolean verbose = false;

  protected MultivariateFunction asMetaF(final DeltaBuffer lessonVector, final TrainingContext trainingContext) {
    List<DeltaFlushBuffer> vector = lessonVector.vector();
    if (isVerbose()) {
      String toString = vector.stream().map(x -> x.toString()).reduce((a, b) -> a + "\n\t" + b).get();
      log.debug(String.format("Optimizing delta vector set: \n\t%s", toString));
    }

    final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
    final double prev = gradientDescentTrainer.getError();
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[vector.size()];

      @Override
      public double value(final double x[]) {
        final int layerCount = vector.size();
        final double[] layerRates = Arrays.copyOf(x, layerCount);
        assert layerRates.length == this.pos.length;
        for (int i = 0; i < layerRates.length; i++) {
          final double prev = this.pos[i];
          final double next = layerRates[i];
          final double adj = next - prev;
          vector.get(i).write(adj);
        }
        for (int i = 0; i < layerRates.length; i++) {
          this.pos[i] = layerRates[i];
        }
        final double calcError = gradientDescentTrainer.calcError(trainingContext, gradientDescentTrainer.evalValidationData(trainingContext));
        final double err = Util.geomMean(calcError);
        if (isVerbose()) {
          DynamicRateTrainer.log.debug(String.format("f[%s] = %s (%s; %s)", Arrays.toString(layerRates), err, calcError, prev - calcError));
        }
        return err;
      }
    };
    return f;
  }

  protected synchronized double calcSieves(final TrainingContext trainingContext) {
    final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
    {
      final NDArray[][] data = gradientDescentTrainer.getConstraintData(trainingContext);
      final List<NNResult> results = gradientDescentTrainer.eval(trainingContext, data);
      final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, data, results.stream().map(x -> x.data).collect(Collectors.toList()));
      updateConstraintSieve(rms);
    }
    {
      final NDArray[][] data = gradientDescentTrainer.getTrainingData(gradientDescentTrainer.getTrainingSet());
      final List<NNResult> list = gradientDescentTrainer.eval(trainingContext, data);
      final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, data, list.stream().map(x1 -> x1.data).collect(Collectors.toList()));
      updateTrainingSieve(rms);
    }
    {
      final List<NDArray> results = gradientDescentTrainer.evalValidationData(trainingContext);
      final NDArray[][] data = gradientDescentTrainer.getValidationData(trainingContext);
      final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, data, results);
      updateValidationSieve(rms);
      return Util.rms(trainingContext, rms, gradientDescentTrainer.getValidationSet());
    }
  }

  protected synchronized boolean calibrate(final TrainingContext trainingContext) {
    synchronized (trainingContext) {
      trainingContext.calibrations.increment();
      final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
      gradientDescentTrainer.setTrainingSet(null);
      gradientDescentTrainer.setValidationSet(null);
      gradientDescentTrainer.setConstraintSet(new int[] {});
      // trainingContext.calcSieves(getInner());
      try {
        double[] key = optimizeRates(trainingContext);
        if(setRates(trainingContext, key)) {
          calcSieves(trainingContext);
          this.lastCalibratedIteration = this.currentIteration;
          final double improvement = -gradientDescentTrainer.step(trainingContext);
          if (isVerbose()) {
            DynamicRateTrainer.log.debug(String.format("Adjusting rates by %s: (%s improvement)", Arrays.toString(key), improvement));
          }
          return true;
        }
      } catch (final Exception e) {
        if (isVerbose()) {
          DynamicRateTrainer.log.debug("Error calibrating", e);
        }
      }
      if (isVerbose()) {
        DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", //
            Arrays.toString(gradientDescentTrainer.getRates()), // 
            gradientDescentTrainer.getError()));
      }
      return false;
    }
  }

  private double[] optimizeRates(final TrainingContext trainingContext) {
    final GradientDescentTrainer inner = getGradientDescentTrainer();
    final NDArray[][] validationSet = inner.getValidationData(trainingContext);
    List<NDArray> evalValidationData = inner.eval(trainingContext, validationSet).stream().map(x1 -> x1.data).collect(Collectors.toList());
    final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, validationSet, evalValidationData);
    final double prev = Util.rms(trainingContext, rms, null);
    // regenDataSieve(trainingContext);
    
    final DeltaBuffer lessonVector = inner.getVector(trainingContext);
    final MultivariateFunction f = asMetaF(lessonVector, trainingContext);
    final int numberOfParameters = lessonVector.vector().size();
    
    final PointValuePair x = new MultivariateOptimizer(f).setMaxRate(getMaxRate()).minimize(numberOfParameters);
    f.value(x.getFirst()); // Leave in optimal state
    // f.value(new double[numberOfParameters]); // Reset to original state
    
    evalValidationData = inner.eval(trainingContext, validationSet).stream().map(x1 -> x1.data).collect(Collectors.toList());
    final double calcError = inner.calcError(trainingContext, evalValidationData);
    inner.setError(calcError);
    if (this.verbose) {
      DynamicRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s", Arrays.toString(x.getKey()), x.getValue(), prev, calcError));
    }
    return x.getKey();
  }

  private boolean setRates(final TrainingContext trainingContext, double[] rates) {
    boolean inBounds = DoubleStream.of(rates).allMatch(r -> getMaxRate() > r) && DoubleStream.of(rates).anyMatch(r -> getMinRate() < r);
    if (inBounds) {
      getGradientDescentTrainer().setRates(rates);
      return true;
    }
    return false;
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

  public int getRecalibrationThreshold() {
    return this.recalibrationThreshold;
  }

  public double getStopError() {
    return this.stopError;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  protected boolean recalibrateWRetry(final TrainingContext trainingContext) {
    int retry = 0;
    while (!calibrate(trainingContext)) {
      DynamicRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
      if (++retry > 0)
        return false;
    }
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
    getGradientDescentTrainer().setVerbose(verbose);
    return this;
  }

  public boolean train(final TrainingContext trainingContext) throws TerminationCondition {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    train(trainingContext, new UniformAdaptiveRateParams(0.1, 1e-8, 1.3, 2.,0.0, java.util.concurrent.TimeUnit.HOURS.toMillis(1)));
    //train2(trainingContext);
    return false;
  }

  private void train(final TrainingContext trainingContext, UniformAdaptiveRateParams params) {
    calcSieves(trainingContext);
    int rateNumber = probeRateCount(trainingContext);
    final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
    double rate = params.startRate;
    RateMonitor linearLearningRate = new RateMonitor(params.terminalETA / 32);
    while (gradientDescentTrainer.getError()>params.terminalLearningRate) {
      double rate1 = rate;
      setRates(trainingContext, IntStream.range(0, rateNumber).mapToDouble(x->rate1).toArray());
      double delta = gradientDescentTrainer.step(trainingContext);
      double projectedEndSeconds = -gradientDescentTrainer.getError()/(linearLearningRate.add(delta)*1000.);
      if (isVerbose()) {
        log.debug(String.format("Projected final convergence time: %.3f sec", projectedEndSeconds));
      }
      if(projectedEndSeconds > params.terminalETA) {
        log.debug(String.format("TERMINAL Projected final convergence time: %.3f sec", projectedEndSeconds));
        break;
      }
      if(0. <= delta) {
        calcSieves(trainingContext);
        rate /= Math.pow(params.alpha, params.beta);
      } else if(0. > delta) {
        rate *= params.alpha;
      } else assert(false);
      if(rate < params.endRate) {
        break;
      }
    }
    if (isVerbose()) {
      DynamicRateTrainer.log.debug("Final network state: " + getGradientDescentTrainer().getNet().toString());
    }
  }

  private int probeRateCount(final TrainingContext trainingContext) {
    final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
    NNResult probe = gradientDescentTrainer.getNet().eval(getGradientDescentTrainer().getTrainingData(trainingContext)[0][0]);
    DeltaBuffer buffer = new DeltaBuffer();
    probe.feedback(new LogNDArray(probe.data.getDims()), buffer);
    int rateNumber = buffer.vector().size();
    return rateNumber;
  }

  public boolean train2(final TrainingContext trainingContext) throws TerminationCondition {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (true) {
      final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
      if (getStopError() > gradientDescentTrainer.getError()) {
        DynamicRateTrainer.log.debug("Target error reached: " + gradientDescentTrainer.getError());
        return false;
      }
      if (this.maxIterations <= this.currentIteration++) {
        DynamicRateTrainer.log.debug("Maximum recalibrations reached: " + this.currentIteration);
        return false;
      }
      if (this.lastCalibratedIteration < this.currentIteration - this.recalibrationInterval) {
        if (isVerbose()) {
          DynamicRateTrainer.log.debug("Recalibrating learning rate due to interation schedule at " + this.currentIteration);
          DynamicRateTrainer.log.debug("Network state: " + getGradientDescentTrainer().getNet().toString());
        }
        if (!recalibrateWRetry(trainingContext))
          return false;
        this.generationsSinceImprovement = 0;
      }
      if (0. != gradientDescentTrainer.step(trainingContext)) {
        this.generationsSinceImprovement = 0;
      } else {
        if (getRecalibrationThreshold() < this.generationsSinceImprovement++) {
          if (isVerbose()) {
            DynamicRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
            DynamicRateTrainer.log.debug("Network state: " + getGradientDescentTrainer().getNet().toString());
          }
          if (!recalibrateWRetry(trainingContext))
            return false;
          this.generationsSinceImprovement = 0;
        }
      }
    }
  }

  protected void updateConstraintSieve(final List<Tuple2<Double, Double>> rms) {
    getGradientDescentTrainer().setConstraintSet(IntStream.range(0, rms.size()).mapToObj(i -> new Tuple2<>(i, rms.get(0))) //
        // .sorted(Comparator.comparing(t ->
        // t.getSecond().getFirst())).limit(50)
        .filter(t -> t.getSecond().getFirst() > 0.8)
        .mapToInt(t -> t.getFirst()).toArray());
  }

  protected void updateTrainingSieve(final List<Tuple2<Double, Double>> rms) {
    getGradientDescentTrainer().setTrainingSet(IntStream.range(0, rms.size()).mapToObj(i -> new Tuple2<>(i, rms.get(0))) //
        .filter(t -> t.getSecond().getFirst() < 0.0)
        //.filter(t -> 1.8 * Math.random() > -0.5 - t.getSecond().getFirst())
        // .sorted(Comparator.comparing(t ->
        // -t.getSecond().getFirst())).limit(100)
        .mapToInt(t -> t.getFirst()).toArray());
  }

  protected void updateValidationSieve(final List<Tuple2<Double, Double>> rms) {
    final List<Tuple2<Integer, Tuple2<Double, Double>>> collect = new ArrayList<>(
        IntStream.range(0, rms.size()).mapToObj(i -> new Tuple2<>(i, rms.get(0))).collect(Collectors.toList()));
    Collections.shuffle(collect);
    getGradientDescentTrainer().setValidationSet(collect.stream() //
        .limit(100)
        // .filter(t -> t.getSecond().getFirst() < -0.3)
        // .filter(t -> 0.5 * Math.random() > -0. - t.getSecond().getFirst())
        // .sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(100)
        // .sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(500)
        .mapToInt(t -> t.getFirst()).toArray());
  }

}
