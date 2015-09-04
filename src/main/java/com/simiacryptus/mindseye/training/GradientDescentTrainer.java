package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.DeltaFlushBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

import groovy.lang.Tuple2;

public class GradientDescentTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);
  
  public static boolean thermalStep(final double prev, final double next, final double temp) {
    if (next < prev) return true;
    if (temp <= 0.) return false;
    final double p = Math.exp(-(next - prev) / (Math.min(next, prev) * temp));
    final boolean step = Math.random() < p;
    return step;
  }
  
  private double error = Double.POSITIVE_INFINITY;
  private PipelineNetwork net = null;
  private double rate = 0.1;
  private double temperature = 0.01;
  private NDArray[][] masterTrainingData = null;
  private boolean verbose = false;
  
  public GradientDescentTrainer() {
  }

  protected double calcError(TrainingContext trainingContext, final List<NDArray> results) {
    final NDArray[][] trainingData = getActiveValidationData(trainingContext);
    final List<Tuple2<Double, Double>> rms = stats(trainingContext, trainingData, results);
    return DynamicRateTrainer.rms(trainingContext,rms, trainingContext.getActiveValidationSet());
  }

  public static List<Tuple2<Double, Double>> stats(TrainingContext trainingContext, final NDArray[][] trainingData, final List<NDArray> results) {
    final List<Tuple2<Double, Double>> rms = IntStream.range(0, results.size()).parallel().mapToObj(sample -> {
      final NDArray actualOutput = results.get(sample);
      final NDArray[] sampleRow = trainingData[sample];
      final NDArray idealOutput = sampleRow[1];
      final double err = actualOutput.rms(idealOutput);
      
      double[] actualOutputData = actualOutput.getData();
      double max = DoubleStream.of(actualOutputData).max().getAsDouble();
      double sum = DoubleStream.of(actualOutputData).sum();
      boolean correct = outputToClassification(actualOutput) == outputToClassification(idealOutput);
      double certianty = (max / sum) * (correct ? 1 : -1);
      return new Tuple2<>(certianty, err * err);
    }).collect(Collectors.toList());
    return rms;
  }
  
  protected List<NNResult> eval(final TrainingContext trainingContext, NDArray[][] trainingData) {
    return Stream.of(trainingData)
        .parallel()
        .map(sample -> {
          final NDArray input = sample[0];
          final NDArray output = sample[1];
          trainingContext.evaluations.increment();
          final NNResult eval = getNet().eval(input);
          assert eval.data.dim() == output.dim();
          return eval;
        }).collect(Collectors.toList());
  }

  protected List<NDArray> evalValidationData(final TrainingContext trainingContext) {
    NDArray[][] validationSet = getActiveValidationData(trainingContext);
    List<NNResult> eval = eval(trainingContext,validationSet);
    return eval.stream().map(x->x.data).collect(Collectors.toList());
  }
  
  public synchronized double getError() {
    return this.error;
  }
  
  public List<NNLayer> getLayers() {
    return getNet().getChildren().stream().distinct().collect(Collectors.toList());
  }
  
  public PipelineNetwork getNet() {
    return this.net;
  }
  
  public double getRate() {
    return this.rate;
  }
  
  public double getTemperature() {
    return this.temperature;
  }
  
  public final NDArray[][] getActiveValidationData(TrainingContext trainingContext) {
    if (null != trainingContext.getActiveValidationSet()) { //
      return IntStream.of(trainingContext.getActiveValidationSet()).mapToObj(i -> masterTrainingData[i]).toArray(i -> new NDArray[i][]);
    }
    return this.masterTrainingData;
  }
  
  public final NDArray[][] getActiveTrainingData(TrainingContext trainingContext) {
    return getTrainingData(trainingContext.getActiveTrainingSet());
  }

  public final NDArray[][] getConstraintData(TrainingContext trainingContext) {
    return getTrainingData(trainingContext.getConstraintSet());
  }

  public NDArray[][] getTrainingData(int[] activeSet) {
    if (null != activeSet) { //
      return IntStream.of(activeSet).mapToObj(i -> masterTrainingData[i]).toArray(i -> new NDArray[i][]);
    }
    return this.masterTrainingData;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public static Integer outputToClassification(NDArray actual) {
    return IntStream.range(0, actual.dim()).mapToObj(o -> o).max(Comparator.comparing(o -> actual.get((int) o))).get();
  }
  
  protected DeltaBuffer prelearn(TrainingContext trainingContext) {
    final DeltaBuffer primary = calcDelta(trainingContext, getActiveTrainingData(trainingContext));
    final DeltaBuffer constraint = calcDelta(trainingContext, getConstraintData(trainingContext)).unitV();
    double dotProductConstraint = primary.dotProduct(constraint);
    if(dotProductConstraint < 0) {
      //log.debug(String.format("Removing component: %s", dotProductConstraint));
      return primary.add(constraint.scale(-dotProductConstraint));
    } else {
      //log.debug(String.format("Preserving component: %s", dotProductConstraint));
      return primary;
    }
  }

  public DeltaBuffer calcDelta(TrainingContext trainingContext, NDArray[][] activeTrainingData) {
    final List<NNResult> netresults = eval(trainingContext, activeTrainingData);
    final DeltaBuffer buffer = new DeltaBuffer();
    IntStream.range(0, activeTrainingData.length)
        .parallel()
        .forEach(sample -> {
          final NDArray idealOutput = activeTrainingData[sample][1];
          final NNResult actualOutput = netresults.get(sample);
          final NDArray delta = actualOutput.delta(idealOutput);
          final LogNDArray logDelta = delta.log().scale(getRate());
          actualOutput.feedback(logDelta, buffer);
        });
    return buffer;
  }
  
  public GradientDescentTrainer setError(final double error) {
    this.error = error;
    return this;
  }
  
  public GradientDescentTrainer setNet(final PipelineNetwork net) {
    this.net = net;
    return this;
  }
  
  public GradientDescentTrainer setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    return this;
  }
  
  public GradientDescentTrainer setTemperature(final double temperature) {
    this.temperature = temperature;
    return this;
  }
  
  public GradientDescentTrainer setMasterTrainingData(final NDArray[][] trainingData) {
    this.masterTrainingData = trainingData;
    return this;
  }
  
  public GradientDescentTrainer setVerbose(final boolean verbose) {
    if (verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }
  
  public synchronized double trainSet(final TrainingContext trainingContext, final double[] rates) {
    assert null != this;
    final double prevError = calcError(trainingContext, evalValidationData(trainingContext));
    setError(prevError);
    if (null == rates) return Double.POSITIVE_INFINITY;
    final DeltaBuffer buffer = prelearn(trainingContext);
    assert rates.length == buffer.map.size();
    final List<DeltaFlushBuffer> deltas = buffer.map.values().stream().collect(Collectors.toList());
    assert rates.length == deltas.size();
    if (null != rates) {
      IntStream.range(0, buffer.map.size()).forEach(i -> deltas.get(i).write(rates[i]));
      final double validationError = calcError(trainingContext, evalValidationData(trainingContext));
      if (prevError == validationError) {
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Static: (%s)", prevError));
        }
      } else if (!GradientDescentTrainer.thermalStep(prevError, validationError, getTemperature())) {
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Reverting delta: (%s -> %s) - %s", prevError, validationError, validationError - prevError));
        }
        IntStream.range(0, buffer.map.size()).forEach(i -> deltas.get(i).write(-rates[i]));
        return prevError;
      } else {
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Validated: (%s)", prevError));
        }
        setError(validationError);
      }
      return validationError;
    } else return prevError;
  }

  public Double step(TrainingContext trainingContext, final double[] rates) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    trainSet(trainingContext, rates);
    trainingContext.gradientSteps.increment();
    if (this.verbose)
    {
      log.debug(String.format("Trained Error: %s (%s) with rate %s*%s in %.03fs",
          getError(), (getError()), getRate(), Arrays.toString(rates),
          (System.currentTimeMillis() - startMs) / 1000.));
    }
    return getError();
  }

  
}
