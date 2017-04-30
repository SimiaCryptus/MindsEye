package com.simiacryptus.mindseye.net.dag;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

public interface DAGNode {
  NNResult get(EvaluationContext buildExeCtx);
  
  DAGNode add(final NNLayer<?> nextHead);

}