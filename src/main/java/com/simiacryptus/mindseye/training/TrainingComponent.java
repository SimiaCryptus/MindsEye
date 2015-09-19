package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

public interface TrainingComponent {

  double getError();

  DAGNetwork getNet();

  double step(TrainingContext trainingContext) throws TerminationCondition;

}
