package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public interface DAGNode {
  NNResult get(EvaluationContext buildExeCtx);
  
  DAGNode add(final NNLayer<?> nextHead);

}