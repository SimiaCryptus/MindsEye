package com.simiacryptus.mindseye.training.dev;

import java.util.List;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.DAGNetwork.DAGNode;
import com.simiacryptus.mindseye.net.DAGNetwork.EvaluationContext;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.TrainingContext;

import groovy.lang.Tuple2;

public class ConstrainedGDTrainer extends GradientDescentTrainer {

  private List<DAGNode> constraintNodes = new java.util.ArrayList<>();

  public ConstrainedGDTrainer() {
    super();
  }

  @Override
  protected DeltaSet calcDelta(final TrainingContext trainingContext, final NDArray[][] data) {
    List<Tuple2<EvaluationContext, Integer>> contexts = initContexts(trainingContext, data, isParallelTraining(), getPrimaryNode());
    DeltaSet primaryDelta = collectVector(getPrimaryNode(), contexts);
    for(DAGNode node : getConstraintNodes()) {
      final DeltaSet constraintDelta = collectVector(node, contexts).unitV();
      double dotProduct = primaryDelta.dotProduct(constraintDelta);
      if(dotProduct<0) {
        primaryDelta = primaryDelta.add(constraintDelta.scale(dotProduct));
      }
    }
    return primaryDelta;
  }

  public DAGNode getPrimaryNode() {
    return super.getPrimaryNode();
  }

  public void setPrimaryNode(DAGNode primaryNode) {
    super.setPrimaryNode(primaryNode);
  }
  
  public List<DAGNode> getConstraintNodes() {
    return constraintNodes;
  }

  public void addConstraintNodes(List<DAGNode> constraintNodes) {
    this.constraintNodes.addAll(constraintNodes);
  }

  public void addConstraintNodes(DAGNode... constraintNodes) {
    java.util.Arrays.stream(constraintNodes).forEach(x->this.constraintNodes.add(x));
  }

}
