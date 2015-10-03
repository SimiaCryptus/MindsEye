package com.simiacryptus.mindseye.test.demo;
/*
Copyright (c) 2010-2011, Advanced Micro Devices, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer. 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution. 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission. 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
If you use the software (in whole or in part), you shall adhere to all applicable U.S., European, and other export
laws, including but not limited to the U.S. Export Administration Regulations ("EAR"), (15 C.F.R. Sections 730 through
774), and E.U. Council Regulation (EC) No 1334/2000 of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR,
you hereby certify that, except pursuant to a license granted by the United States Department of Commerce Bureau of 
Industry and Security or as otherwise permitted pursuant to a License Exception under the U.S. Export Administration 
Regulations ("EAR"), you will not (1) export, re-export or release to a national of a country in Country Groups D:1,
E:1 or E:2 any restricted technology, software, or source code you receive hereunder, or (2) export to Country Groups
D:1, E:1 or E:2 the direct product of such technology or software, if such foreign produced direct product is subject
to national security controls as identified on the Commerce Control List (currently found in Supplement 1 to Part 774
of EAR).  For the most current Country Group listings, or for additional information about the EAR or your obligations
under those regulations, please refer to the U.S. Bureau of Industry and Security's website at http://www.bis.doc.gov/. 
*/

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.device.Device;
import com.amd.aparapi.device.OpenCLDevice;
import com.amd.aparapi.internal.opencl.OpenCLPlatform;
import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.util.Util;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.Range;

public class DeltaValidationTest  {
  private static final Logger log = LoggerFactory.getLogger(DeltaValidationTest.class);
  
  double deltaFactor = 1e-6;

  @org.junit.Test
  public void testDenseSynapseLayer1(){
    NDArray outputPrototype = new NDArray(2);
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(()->Util.R.get().nextGaussian());
    test(outputPrototype, inputPrototype, component);
  }


  public void test(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component) {
    NDArray measuredGradient = measureFeedbackGradient(outputPrototype, inputPrototype, component);
    NDArray implementedGradient = getFeedbackGradient(outputPrototype, inputPrototype, component);
    
    log.debug(String.format("%s",measuredGradient));
    log.debug(String.format("%s",implementedGradient));
    log.debug(String.format("%s",measuredGradient.minus(implementedGradient)));
  }


  @org.junit.Test
  public void testBiasLayer(){
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new BiasLayer(outputPrototype.getDims()).setWeights(i->Util.R.get().nextGaussian());
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testBiasLayer2(){
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NDArray outputPrototype = inputPrototype.copy();
    NNLayer<?> component = new SoftmaxActivationLayer();
    test(outputPrototype, inputPrototype, component);
  }


  public NDArray measureFeedbackGradient(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component) {
    NDArray measuredGradient = new NDArray(inputPrototype.dim(), outputPrototype.dim());
    NDArray baseOutput = component.eval(new EvaluationContext(), inputPrototype).data;
    for(int i=0;i<inputPrototype.dim();i++) {
      NDArray inputProbe = inputPrototype.copy();
      inputProbe.add(i, deltaFactor * 1);
      NDArray evalProbe = component.eval(new EvaluationContext(), inputProbe).data;
      NDArray delta = evalProbe.minus(baseOutput).scale(1./deltaFactor);
      for(int j=0;j<delta.dim();j++){
        measuredGradient.set(new int[]{i,j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }


  public NDArray getFeedbackGradient(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component) {
    NDArray implementedGradient = new NDArray(inputPrototype.dim(), outputPrototype.dim());
    for(int j=0;j<outputPrototype.dim();j++){
      int j_ = j;
      EvaluationContext evaluationContext = new EvaluationContext();
      component.eval(evaluationContext, new NNResult(evaluationContext, inputPrototype) {
        
        @Override
        public boolean isAlive() {
          return true;
        }
        
        @Override
        public void feedback(NDArray data, DeltaBuffer buffer) {
          for(int i=0;i<inputPrototype.dim();i++) {
            implementedGradient.set(new int[]{i,j_}, data.getData()[i]);
          }
        }
      }).feedback(new NDArray(outputPrototype.getDims()).set(j, 1), new DeltaBuffer());
    }
    return implementedGradient;
  }

  
}
