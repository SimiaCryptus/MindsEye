package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.dev.ToeplitzSynapseLayerJBLAS;

import java.util.Collection;
import java.util.stream.Collectors;

public class L12Normalized implements OrientationStrategy {
    public final OrientationStrategy inner;
    private double factor_L1 = 0.01;
    private double factor_L2 = 0.0;

    public L12Normalized() {
        this(new LBFGS());
    }

    public L12Normalized(OrientationStrategy inner) {
        this.inner = inner;
    }

    public double directionFilter(double unitDotProduct) {
        return unitDotProduct<0?0:1;
    }

    @Override
    public DeltaSet orient(Trainable.PointSample measurement, TrainingMonitor monitor) {
        DeltaSet primaryVector = inner.orient(measurement, monitor);
        DeltaSet l1Vector = new DeltaSet();
        DeltaSet l2Vector = new DeltaSet();
        for(NNLayer layer : getLayers(primaryVector.map.keySet())) {
            double[] weights = primaryVector.map.get(layer).target;
            double[] delta = primaryVector.get(layer, weights).delta;
            for(int i=0;i<delta.length;i++) {
                double[] l1delta = l1Vector.get(layer, weights).delta;
                double[] l2delta = l2Vector.get(layer, weights).delta;
                l1delta[i] = weights[i]<0?-1:1;
                l2delta[i] = weights[i];
            }
        }
        l1Vector = l1Vector.unit();
        double l1dot = l1Vector.dot(primaryVector);
        l1Vector = l1Vector.scale(directionFilter(l1dot));
        l2Vector = l2Vector.unit();
        double l2dot = l2Vector.dot(primaryVector);
        l2Vector = l2Vector.scale(directionFilter(l2dot));
        return primaryVector.add(l1Vector.scale(-factor_L1)).add(l2Vector.scale(-factor_L2));
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

    public L12Normalized setFactor_L1(double factor_L1) {
        this.factor_L1 = factor_L1;
        return this;
    }

    public double getFactor_L2() {
        return factor_L2;
    }

    public L12Normalized setFactor_L2(double factor_L2) {
        this.factor_L2 = factor_L2;
        return this;
    }
}
