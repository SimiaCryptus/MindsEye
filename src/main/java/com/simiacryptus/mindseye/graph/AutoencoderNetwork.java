package com.simiacryptus.mindseye.graph;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.graph.dag.DAGNode;

public class AutoencoderNetwork extends PipelineNetwork {

    public final DAGNode encoder;
    public final DAGNode decoder;

    public AutoencoderNetwork(final NNLayer encoder, final NNLayer decoder) {
        super(1);
        this.encoder = add(encoder);
        this.decoder = add(decoder);
    }

}
