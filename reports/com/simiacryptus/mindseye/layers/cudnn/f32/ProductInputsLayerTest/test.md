# ProductInputsLayer
## ProductInputsLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
      "id": "370a9587-74a1-4959-b406-fa45000003db",
      "isFrozen": false,
      "name": "ProductInputsLayer/370a9587-74a1-4959-b406-fa45000003db"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[
    	[ [ -1.536, 0.876 ], [ 0.972, -1.936 ] ],
    	[ [ 0.252, -0.596 ], [ -1.732, -0.652 ] ]
    ],
    [
    	[ [ 0.988, 1.704 ], [ -0.308, -0.64 ] ],
    	[ [ 1.856, 0.144 ], [ 1.64, 0.408 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.5175679922103882, 1.4927040338516235 ], [ -0.2993760108947754, 1.2390400171279907 ] ],
    	[ [ 0.4677119851112366, -0.08582399785518646 ], [ -2.840479850769043, -0.26601600646972656 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.536, 0.876 ], [ 0.972, -1.936 ] ],
    	[ [ 0.252, -0.596 ], [ -1.732, -0.652 ] ]
    ],
    [
    	[ [ 0.988, 1.704 ], [ -0.308, -0.64 ] ],
    	[ [ 1.856, 0.144 ], [ 1.64, 0.408 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.045886347037843514, negative=5, min=-0.652, max=-0.652, mean=-0.544, count=8.0, positive=3, stdDev=1.0779443399359727, zeros=0},
    {meanExponent=-0.15332478742271105, negative=2, min=0.408, max=0.408, mean=0.7240000000000001, count=8.0, positive=6, stdDev=0.9022438694721066, zeros=0}
    Output: [
    	[ [ -1.5175679922103882, 1.4927040338516235 ], [ -0.2993760108947754, 1.2390400171279907 ] ],
    	[ [ 0.4677119851112366, -0.08582399785518646 ], [ -2.840479850769043, -0.26601600646972656 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.19921113539863788, negative=5, min=-0.26601600646972656, max=-0.26601600646972656, mean=-0.2262259777635336, count=8.0, positive=3, stdDev=1.3281476347449674, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.536, 0.876 ], [ 0.972, -1.936 ] ],
    	[ [ 0.252, -0.596 ], [ -1.732, -0.652 ] ]
    ]
    Value Statistics: {meanExponent=-0.045886347037843514, negative=5, min=-0.652, max=-0.652, mean=-0.544, count=8.0, positive=3, stdDev=1.0779443399359727, zeros=0}
    Implemented Feedback: [ [ 0.9879999756813049, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.8559999465942383, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.30799999833106995, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.6399999856948853, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.7039999961853027, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.14399999380111694, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6399999856948853, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40799999237060547 ] ]
    Implemented Statistics: {meanExponent=-0.15332479577569585, negative=2, min=0.40799999237060547, max=0.40799999237060547, mean=0.09049999853596091, count=64.0, positive=6, stdDev=0.3988574244677062, zeros=56}
    Measured Feedback: [ [ 0.9882450103759766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.855790615081787, 0.0, 0.0, 0.0, 0.0, 0.
```
...[skipping 1884 bytes](etc/1.txt)...
```
    59999871253967, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.5960000157356262, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.9359999895095825, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6520000100135803 ] ]
    Implemented Statistics: {meanExponent=-0.04588634456725603, negative=5, min=-0.6520000100135803, max=-0.6520000100135803, mean=-0.06800000043585896, count=64.0, positive=3, stdDev=0.42144216667187095, zeros=56}
    Measured Feedback: [ [ -1.5366077423095703, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.25212764739990234, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9718537330627441, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.7333030700683594, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.8749961853027344, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.5960464477539062, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.9359588623046875, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6520748138427734 ] ]
    Measured Statistics: {meanExponent=-0.04585766045204819, negative=5, min=-0.6520748138427734, max=-0.6520748138427734, mean=-0.06804708391427994, count=64.0, positive=3, stdDev=0.42151610524489597, zeros=56}
    Feedback Error: [ [ -6.077289581298828E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.2764334678649902E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.462697982788086E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.0013030767440795898, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.0010038018226623535, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -4.64320182800293E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.112720489501953E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.480382919311523E-5 ] ]
    Error Statistics: {meanExponent=-3.709206082939218, negative=6, min=-7.480382919311523E-5, max=-7.480382919311523E-5, mean=-4.7083478420972824E-5, count=64.0, positive=2, stdDev=2.1579118974392682E-4, zeros=56}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.8642e-05 +- 1.6371e-04 [0.0000e+00 - 1.3031e-03] (128#)
    relativeTol: 1.5211e-04 +- 1.4482e-04 [7.5967e-06 - 5.7327e-04] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.8642e-05 +- 1.6371e-04 [0.0000e+00 - 1.3031e-03] (128#), relativeTol=1.5211e-04 +- 1.4482e-04 [7.5967e-06 - 5.7327e-04] (16#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 4.1776 +- 0.7491 [3.4340 - 8.0336]
    Learning performance: 0.5900 +- 0.2399 [0.2850 - 2.2257]
    
```

