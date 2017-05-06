package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

public interface TrainingComponent {

  public static class TrainingStep {
    private boolean changed;
    private double startError;
    double testError;

    public TrainingStep(final double startError, final double endError, final boolean changed) {
      super();
      setStartError(startError);
      this.testError = endError;
      setChanged(changed);
    }

    public double finalError() {
      return isChanged() ? testError : getStartError();
    }

    public double getStartError() {
      return startError;
    }

    public double improvement() {
      return getStartError() - testError;
    }

    public boolean isChanged() {
      return changed;
    }

    public void setChanged(final boolean changed) {
      this.changed = changed;
    }

    public void setStartError(final double startError) {
      this.startError = startError;
    }

    @Override
    public String toString() {
      return "TrainingStep [changed=" + changed + ", startError=" + startError + ", testError=" + testError + "]";
    }
    
    
  }

  Tensor[][] getTrainingData();

  double getError();

  DAGNetwork getNet();

  void reset();

  TrainingComponent setData(Tensor[][] data);

  TrainingStep step(TrainingContext trainingContext) throws TerminationCondition;

}
