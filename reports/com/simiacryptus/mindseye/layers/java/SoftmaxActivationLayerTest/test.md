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
      "id": "f4569375-56fe-4e46-925c-95f400000a76",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/f4569375-56fe-4e46-925c-95f400000a76"
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
    [[ -1.7, -0.796, 1.316, 1.604 ]]
    --------------------
    Output: 
    [ 0.019569372772867098, 0.04832580729320872, 0.399400922403988, 0.5327038975299362 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.7, -0.796, 1.316, 1.604 ]
    Output: [ 0.019569372772867098, 0.04832580729320872, 0.399400922403988, 0.5327038975299362 ]
    Measured: [ [ 0.01918733422451302, -9.457511734867197E-4, -0.007816401053628574, -0.010425181997675281 ], [ -9.457484537178651E-4, 0.045992500966943184, -0.019302243824959042, -0.025744508688196888 ], [ -0.007816104159140413, -0.019301566164808825, 0.2398822385801802, -0.21276456825702894 ], [ -0.010424647046945734, -0.02574326168437724, -0.2127617320507591, 0.24892964078149227 ] ]
    Implemented: [ [ 0.019186412422143667, -9.457057374705408E-4, -0.007816025536350607, -0.010424681148322517 ], [ -9.457057374705408E-4, 0.04599042364266837, -0.019301372008824935, -0.0257433458963729 ], [ -0.007816025536350607, -0.019301372008824935, 0.23987982558683157, -0.21276242804165604 ], [ -0.010424681148322517, -0.0257433458963729, -0.21276242804165604, 0.24893045508635145 ] ]
    Error: [ [ 9.218023693524646E-7, -4.5436016178840646E-8, -3.7551727796736323E-7, -5.008493527639685E-7 ], [ -4.271624732426517E-8, 2.0773242748114984E-6, -8.718161341077224E-7, -1.1627918239873192E-6 ], [ -7.862278980641513E-8, -1.941559838909701E-7, 2.4129933486327637E-6, -2.140215372897769E-6 ], [ 3.4101376783154946E-8, 8.42119956614984E-8, 6.95990896942833E-7, -8.143048591813251E-7 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.7830e-07 +- 7.7393e-07 [3.4101e-08 - 2.4130e-06] (16#)
    relativeTol: 1.3318e-05 +- 1.0070e-05 [1.6356e-06 - 2.4022e-05] (16#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2912 +- 0.0703 [0.2166 - 0.5101]
    Learning performance: 0.0026 +- 0.0022 [0.0000 - 0.0142]
    
```

