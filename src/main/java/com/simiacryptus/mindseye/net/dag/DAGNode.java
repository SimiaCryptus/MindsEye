package com.simiacryptus.mindseye.net.dag;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.UUID;

public interface DAGNode {

  UUID getId();

  NNLayer getLayer();

  NNResult get(EvaluationContext buildExeCtx);

  default DAGNode[] getInputs() {
    return new DAGNode[]{};
  }

    JsonObject toJson();

}