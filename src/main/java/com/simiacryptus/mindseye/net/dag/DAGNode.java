package com.simiacryptus.mindseye.net.dag;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.UUID;

public interface DAGNode {

  UUID getId();

  NNResult get(EvaluationContext buildExeCtx);
  
  DAGNode add(final NNLayer<?> nextHead);

  default DAGNode[] getInputs() {
    return new DAGNode[]{};
  };

}