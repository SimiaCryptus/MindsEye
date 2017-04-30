package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class GradientDescentTrainer extends StochasticTrainer implements RateTrainingComponent {

  private static class RevertableStep {
    public double finalError;
    public double prevError;

    public RevertableStep(final double prevError, final double finalError) {
      super();
      this.prevError = prevError;
      this.finalError = finalError;
    }

  }

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  private double rate = 0.1;

  @Override
  public TrainingStep step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    final Tensor[][] data = selectTrainingData();
    if (data.length == 0) {
      return new TrainingStep(Double.NaN, Double.NaN, false);
    }
    final double prevError = summarize(trainingContext, data).rms;
    final EvaluationContext contexts = createBatchExeContext(trainingContext, data);
    final DeltaSet buffer = new DeltaSet();
    getPrimaryNode().get(contexts).accumulate(buffer);
    buffer.vector().stream().forEach(d -> d.write(-this.rate));
    final double finalError = summarize(trainingContext, data).rms;
    if (prevError == finalError) {
      if (this.isVerbose()) {
        log.debug(String.format("Static: (%s)", prevError));
      }
      setError(finalError);
      trainingContext.gradientSteps.increment();
      return new TrainingStep(prevError, finalError, false);
    } else {
      setError(finalError);
      trainingContext.gradientSteps.increment();
      if (this.isVerbose()) {
        log.debug(String.format("Step Complete in %.03f  - Error %s with rate %s and %s items - %s", //
                (System.currentTimeMillis() - startMs) / 1000., finalError, getRate(), Math.min(getTrainingSize(), getTrainingData().length), trainingContext));
      }
      return new TrainingStep(prevError, finalError, true);
    }
  }

  @Override
  public double getRate() {
    return this.rate;
  }

  @Override
  public void reset() {
    setError(Double.NaN);
  }

  @Override
  public GradientDescentTrainer setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    setError(Double.NaN);
    return this;
  }

}
