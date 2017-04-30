package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IterativeTrainer extends DelegateTrainer<TrainingComponent> {
  private static final Logger log = LoggerFactory.getLogger(IterativeTrainer.class);

  protected IterativeTrainer() {
    this(new LineSearchTrainer());
  }

  public IterativeTrainer(final TrainingComponent inner) {
    super(inner);
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) {
    double prevError = getError();
    while (!Double.isFinite(this.inner.getError()) || this.inner.getError() > trainingContext.terminalErr) {
      final TrainingStep step = this.inner.step(trainingContext);
      if (!Double.isFinite(prevError)) {
        prevError = step.getStartError();
      }
      if (trainingContext.timeout < System.currentTimeMillis()) {
        log.debug(String.format("TIMEOUT; current err: %s", step));
        break;
      }
      if (step.finalError() <= trainingContext.terminalErr) {
        if (isVerbose()) {
          log.debug(String.format("TERMINAL Final err: %s", step));
        }
      }
    }
    if (isVerbose()) {
      String string = getNet().toString();
      if (string.length() > 1024) {
        string = string.substring(0, 1924);
      }
      IterativeTrainer.log.debug("Final network state: " + string);
    }
    final double endError = getError();
    return new TrainingStep(prevError, endError, true);
  }
}
