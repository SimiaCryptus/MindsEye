package com.simiacryptus.mindseye.training;

import java.util.List;
import java.util.stream.Collectors;
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

public class GradientDescentTrainer extends SupervisedTrainingParameters {

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  private double error = Double.POSITIVE_INFINITY;
  private double rate = 0.5;
  private boolean verbose = false;

  public GradientDescentTrainer() {
  }

  public GradientDescentTrainer set(final PipelineNetwork net, final NDArray[][] data) {
    this.setNet(net);
    this.setTrainingData(data);
    return this;
  }

  protected double calcError(final List<NNResult> result) {
    final SupervisedTrainingParameters currentNet = this;
    List<Double> rms = IntStream.range(0, result.size()).parallel().mapToObj(sample -> {
      final NNResult eval = result.get(sample);
      NDArray[][] trainingData = currentNet.getTrainingData();
      NDArray[] sampleRow = trainingData[sample];
      NDArray expected = sampleRow[1];
      final NDArray preset = expected;
      final NDArray output = preset;
      final double err = eval.data.rms(output);
      return err * err;
    }).collect(Collectors.toList());
    return Math.sqrt(rms.stream().mapToDouble(x -> x).average().getAsDouble());
  }

  public double error(TrainingContext trainingContext) {
    final double error = getError();
    if (!Double.isFinite(error)) {
      trainSet(trainingContext,null);
      return error(trainingContext);
    }
    return error;
  }

  protected List<NNResult> evalTrainingData(TrainingContext trainingContext) {
    SupervisedTrainingParameters params = this;
    return Stream.of(params.getTrainingData())
        .parallel()
        .map(sample -> {
          final NDArray input = sample[0];
          final NDArray output = sample[1];
          trainingContext.evaluations.increment();
          final NNResult eval = params.getNet().eval(input);
          assert eval.data.dim() == output.dim();
          return eval;
        }).collect(Collectors.toList());
  }

  public synchronized double getError() {
    return this.error;
  }

  public List<NNLayer> getLayers() {
    SupervisedTrainingParameters x = this;
    return x.getNet().getChildren().stream().distinct().collect(Collectors.toList());
  }

  public double getRate() {
    return this.rate;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  protected DeltaBuffer learn(final List<NNResult> netresults) {
    final DeltaBuffer buffer = new DeltaBuffer();
    // Apply corrections
    {
      final SupervisedTrainingParameters currentNet = this;
      IntStream.range(0, netresults.size())
      .parallel()
      .forEach(sample -> {
        final NNResult actualOutput = netresults.get(sample);
        final NDArray idealOutput = currentNet.getTrainingData()[sample][1];
        final NDArray delta = actualOutput.delta(idealOutput);
        LogNDArray logDelta = delta.log().scale(getRate());
        actualOutput.feedback(logDelta, buffer);
      });
    };
    return buffer;
  }

  public GradientDescentTrainer setError(final double error) {
    this.error = error;
    return this;
  }

  public GradientDescentTrainer setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    return this;
  }

  public GradientDescentTrainer setVerbose(final boolean verbose) {
    if (verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }

  public synchronized double trainSet(TrainingContext trainingContext, final double[] rates) {
    assert null != this;
    final List<NNResult> results = evalTrainingData(trainingContext);
    double prevError = calcError(results);
    setError(prevError);
    DeltaBuffer buffer = learn(results);
    if(null==rates) return Double.POSITIVE_INFINITY;
    assert(rates.length==buffer.map.size());
    final List<DeltaFlushBuffer> deltas = buffer.map.values().stream().collect(Collectors.toList());
    assert(rates.length==deltas.size());
    if (null != rates) {
      IntStream.range(0, buffer.map.size()).forEach(i -> deltas.get(i).write(rates[i]));
      final double validationError = calcError(evalTrainingData(trainingContext));
      if(prevError == validationError) {
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Static: (%s)", (prevError)));
        }
      } else if (!thermalStep(prevError, validationError, getTemperature())) {
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Reverting delta: (%s -> %s) - %s", prevError, validationError, validationError-prevError));
        }
        IntStream.range(0, buffer.map.size()).forEach(i -> deltas.get(i).write(-rates[i]));
        return prevError;
      } else{
        if (this.verbose) {
          GradientDescentTrainer.log.debug(String.format("Validated: (%s)", (prevError)));
        }
        setError(validationError);
      }
      return validationError;
    } else {
      return prevError;
    }
  }
  private double temperature = 0.00;

  public static boolean thermalStep(final double prev, final double next, double temp) {
    if(next<prev) return true;
    if(temp<=0.) return false;
    double p = Math.exp(-(next-prev)/(Math.min(next,prev)*temp));
    boolean step = Math.random() < p;
    return step;
  }

  public double getTemperature() {
    return temperature;
  }

  public GradientDescentTrainer setTemperature(double temperature) {
    this.temperature = temperature;
    return this;
  }

}
