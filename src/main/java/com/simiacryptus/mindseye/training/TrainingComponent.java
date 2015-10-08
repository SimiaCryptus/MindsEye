package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

public interface TrainingComponent {

  public static class TrainingStep {
    private double startError;
    double testError;
    private boolean changed;
    public TrainingStep(double startError, double endError, boolean changed) {
      super();
      this.setStartError(startError);
      this.testError = endError;
      this.setChanged(changed);
    }
    public double improvement() {
      return getStartError() - testError;
    }
    public double finalError() {
      return isChanged()?testError:getStartError();
    }
    public double getStartError() {
      return startError;
    }
    public void setStartError(double startError) {
      this.startError = startError;
    }
    public boolean isChanged() {
      return changed;
    }
    public void setChanged(boolean changed) {
      this.changed = changed;
    }
  }
  
  double getError();

  DAGNetwork getNet();

  TrainingStep step(TrainingContext trainingContext) throws TerminationCondition;

  NDArray[][] getData();

  void reset();

}
