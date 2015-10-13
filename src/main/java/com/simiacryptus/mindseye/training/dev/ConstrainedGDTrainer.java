package com.simiacryptus.mindseye.training.dev;

import java.util.List;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.TrainingContext;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.net.DAGNetwork.DAGNode;
import com.simiacryptus.mindseye.net.DAGNetwork.EvaluationContext;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;

public class ConstrainedGDTrainer extends GradientDescentTrainer {

  private List<DAGNode> constraintNodes = new java.util.ArrayList<>();

  public ConstrainedGDTrainer() {
    super();
  }

  public void addConstraintNodes(final DAGNode... constraintNodes) {
    java.util.Arrays.stream(constraintNodes).forEach(x -> this.constraintNodes.add(x));
  }

  public void addConstraintNodes(final List<DAGNode> constraintNodes) {
    this.constraintNodes.addAll(constraintNodes);
  }

  @Override
  protected DeltaSet calcDelta(final TrainingContext trainingContext, final NDArray[][] data) {
    final EvaluationContext contexts = initContexts(trainingContext, data, getPrimaryNode());
    DeltaSet primaryDelta = collectVector(getPrimaryNode(), contexts);
    for (final DAGNode node : getConstraintNodes()) {
      final DeltaSet constraintDelta = collectVector(node, contexts).unitV();
      final double dotProduct = primaryDelta.dotProduct(constraintDelta);
      if (dotProduct < 0) {
        primaryDelta = primaryDelta.add(constraintDelta.scale(dotProduct));
      }
    }
    return primaryDelta;
  }

  public List<DAGNode> getConstraintNodes() {
    return this.constraintNodes;
  }

  @Override
  public DAGNode getPrimaryNode() {
    return super.getPrimaryNode();
  }

  @Override
  public void setPrimaryNode(final DAGNode primaryNode) {
    super.setPrimaryNode(primaryNode);
  }

}
