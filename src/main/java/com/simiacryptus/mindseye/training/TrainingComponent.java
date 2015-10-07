package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

public interface TrainingComponent {

  public static class TrainingStep {
    double startError;
    double testError;
    boolean changed;
    public TrainingStep(double startError, double endError, boolean changed) {
      super();
      this.startError = startError;
      this.testError = endError;
      this.changed = changed;
    }
    public double improvement() {
      return startError - testError;
    }
    public double finalError() {
      return changed?testError:startError;
    }
  }
  
  double getError();

  DAGNetwork getNet();

  TrainingStep step(TrainingContext trainingContext) throws TerminationCondition;

  NDArray[][] getData();

  void reset();

}
