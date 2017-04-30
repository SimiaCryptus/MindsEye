package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.core.delta.DeltaBuffer;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.net.EvaluationContext;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class GradientDescentTrainer extends StochasticTrainer implements RateTrainingComponent {

  private abstract static class RevertableStep {
    public double finalError;
    public double prevError;

    public RevertableStep(final double prevError, final double finalError) {
      super();
      this.prevError = prevError;
      this.finalError = finalError;
    }

    public abstract void revert();

  }

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  private double rate = 0.1;

  @Override
  public TrainingStep step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    final Tensor[][] data = selectTrainingData();
    RevertableStep stepResult;
    if (data.length == 0) {
      stepResult = new RevertableStep(Double.NaN, Double.NaN) {
        @Override
        public void revert() {
        }
      };
    } else {
      final double prevError = validate(trainingContext, data).rms;
      final EvaluationContext contexts = createBatchExeContext(trainingContext, data);
      final DeltaSet buffer = new DeltaSet();
      getPrimaryNode().get(contexts).accumulate(buffer);
      final List<DeltaBuffer> deltas = buffer.scale(-getRate()).vector();
      deltas.stream().forEach(d -> d.write(this.rate));
      final double validationError = validate(trainingContext, data).rms;
      stepResult = new RevertableStep(prevError, validationError) {
        @Override
        public void revert() {
          deltas.stream().forEach(d -> d.write(-GradientDescentTrainer.this.rate));
          validate(trainingContext, data);
        }
      };
    }
    if (stepResult.prevError == stepResult.finalError) {
      if (this.isVerbose()) {
        log.debug(String.format("Static: (%s)", stepResult.prevError));
      }
      setError(stepResult.finalError);
      trainingContext.gradientSteps.increment();
      return new TrainingStep(stepResult.prevError, stepResult.finalError, false);
    } else {
      setError(stepResult.finalError);
      trainingContext.gradientSteps.increment();
      if (this.isVerbose()) {
        log.debug(String.format("Step Complete in %.03f  - Error %s with rate %s and %s items - %s", //
                (System.currentTimeMillis() - startMs) / 1000., stepResult.finalError, getRate(), Math.min(getTrainingSize(), getTrainingData().length), trainingContext));
      }
      return new TrainingStep(stepResult.prevError, stepResult.finalError, true);
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
