# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f9a",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f9a"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -1.828, 0.636, -1.504, -0.628 ]]
    --------------------
    Output: 
    [ 0.057291852693873525, 0.6732780717951974, 0.07921442584707011, 0.19021564966385895 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: SoftmaxActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f9a
    Inputs: [ -1.828, 0.636, -1.504, -0.628 ]
    output=[ 0.057291852693873525, 0.6732780717951974, 0.07921442584707011, 0.19021564966385895 ]
    measured/actual: [ [ 0.054011887413996096, -0.03857505582849363, -0.004538542137944024, -0.01089828944755844 ], [ -0.038572679699269696, 0.21997089803882375, -0.05333241171426306, -0.12806580662438893 ], [ -0.004538532187986499, -0.05333558013242978, 0.07294256984230474, -0.015068457521749679 ], [ -0.010898144578996849, -0.12807199319397888, -0.015068290254716121, 0.15403842802769185 ] ]
    implemented/expected: [ [ 0.054009496308777015, -0.038573348111305646, -0.004538341216860108, -0.01089780698061126 ], [ -0.038573348111305646, 0.21997470983493844, -0.05333333589267901, -0.12806802583095372 ], [ -0.004538341216860108, -0.05333333589267901, 0.07293950058478915, -0.015067823475250023 ], [ -0.01089780698061126, -0.12806802583095372, -0.015067823475250023, 0.154033656286815 ] ]
    error: [ [ 2.391105219080869E-6, -1.7077171879861797E-6, -2.0092108391541602E-7, -4.824669471801407E-7 ], [ 6.684120359493062E-7, -3.811796114694399E-6, 9.241784159480515E-7, 2.219206564790799E-6 ], [ -1.909711263908473E-7, -2.244239750773258E-6, 3.069257515586621E-6, -6.340464996559131E-7 ], [ -3.3759838558890254E-7, -3.9673630251590986E-6, -4.6677946609861853E-7, 4.77174087684662E-6 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7555e-06 +- 1.4597e-06 [1.9097e-07 - 4.7717e-06] (16#)
    relativeTol: 1.6832e-05 +- 5.3466e-06 [8.6642e-06 - 2.2135e-05] (16#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2885 +- 0.0643 [0.2194 - 0.5158]
    Learning performance: 0.0018 +- 0.0014 [0.0000 - 0.0029]
    
```

