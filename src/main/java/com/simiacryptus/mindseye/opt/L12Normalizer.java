package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.dev.ToeplitzSynapseLayerJBLAS;

import java.util.Collection;
import java.util.stream.Collectors;

public class L12Normalizer implements Trainable {
    public final Trainable inner;
    private double factor_L1 = 0.0001;
    private double factor_L2 = 0.0;

    public L12Normalizer(Trainable inner) {
        this.inner = inner;
    }

    @Override
    public PointSample measure() {
        PointSample innerMeasure = inner.measure();
        DeltaSet normalizationVector = new DeltaSet();
        for(NNLayer layer : getLayers(innerMeasure.delta.map.keySet())) {
            double[] weights = innerMeasure.delta.map.get(layer).target;
            double[] delta = normalizationVector.get(layer, weights).delta;
            for(int i=0;i<delta.length;i++) {
                delta[i] -= factor_L1 * (weights[i]<0?-1.0:1.0) + factor_L2 * weights[i];
            }
            assert(null != delta);
        }
        return new PointSample(
                innerMeasure.delta.add(normalizationVector),
                innerMeasure.weights,
                innerMeasure.value);
    }

    @Override
    public void resetToFull() {
        inner.resetToFull();
    }

    @Override
    public void resetSampling() {
        inner.resetSampling();
    }

    public Collection<NNLayer> getLayers(Collection<NNLayer> layers) {
        return layers.stream()
                .filter(layer->{
                    if(layer instanceof DenseSynapseLayerJBLAS) return true;
                    if(layer instanceof ToeplitzSynapseLayerJBLAS) return true;
                    if(layer instanceof DenseSynapseLayer) return true;
                    return false;
                })
                .collect(Collectors.toList());
    }

    public double getFactor_L1() {
        return factor_L1;
    }

    public L12Normalizer setFactor_L1(double factor_L1) {
        this.factor_L1 = factor_L1;
        return this;
    }

    public double getFactor_L2() {
        return factor_L2;
    }

    public L12Normalizer setFactor_L2(double factor_L2) {
        this.factor_L2 = factor_L2;
        return this;
    }
}
