# SimpleConvolutionLayer
## MultiBand
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "2d93a829-d8b6-4d84-aba4-810000000001",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/2d93a829-d8b6-4d84-aba4-810000000001",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          1.392,
          -0.952,
          -1.472,
          -0.552,
          1.84,
          -1.128,
          0.88,
          -1.892,
          1.712
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ 0.98, 0.604, -1.312 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.7204160690307617, 2.0503361225128174, -2.5265119075775146 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.81 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "2d93a829-d8b6-4d84-aba4-810000000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/2d93a829-d8b6-4d84-aba4-810000000002",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          1.392,
          -0.552,
          0.88,
          -0.952,
          1.84,
          -1.892,
          -1.472,
          -1.128,
          1.712
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ 0.98, 0.604, -1.312 ] ]
    ]
    Error: [
    	[ [ 6.903076199549218E-8, 1.2251281722441831E-7, 9.242248522056684E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 9.4655e-08 +- 2.1891e-08 [6.9031e-08 - 1.2251e-07] (3#)
    relativeTol: 2.0285e-08 +- 7.1576e-09 [1.2688e-08 - 2.9876e-08] (3#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.12 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ 0.98, 0.604, -1.312 ] ]
    ]
    Output: [
    	[ [ 2.7204160690307617, 2.0503361225128174, -2.5265119075775146 ] ]
    ]
    Measured: [ [ 1.392364501953125, -0.553131103515625, 0.8797645568847656 ], [ -0.95367431640625, 1.8405914306640625, -1.8930435180664062 ], [ -1.4734268188476562, -1.1301040649414062, 1.7118453979492188 ] ]
    Implemented: [ [ 1.3919999599456787, -0.5519999861717224, 0.8799999952316284 ], [ -0.9520000219345093, 1.840000033378601, -1.8919999599456787 ], [ -1.472000002861023, -1.128000020980835, 1.7120000123977661 ] ]
    Error: [ [ 3.6454200744628906E-4, -0.0011311173439025879, -2.3543834686279297E-4 ], [ -0.0016742944717407227, 5.913972854614258E-4, -0.001043558120727539 ], [ -0.0014268159866333008, -0.002104043960571289, -1.5461444854736328E-4 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ 0.98, 0.604, -1.312 ] ]
    ]
    Outputs: [
    	[ [ 2.7204160690307617, 2.0503361225128174, -2.5265119075775146 ] ]
    ]
    Measured Gradient: [ [ 0.9799003601074219, 0.0, 0.0 ], [ 0.6031990051269531, 0.0, 0.0 ], [ -1.3136863708496094, 0.0, 0.0 ], [ 0.0, 0.9799003601074219, 0.0 ], [ 0.0, 0.6031990051269531, 0.0 ], [ 0.0, -1.3136863708496094, 0.0 ], [ 0.0, 0.0, 0.9799003601074219 ], [ 0.0, 0.0, 0.6031990051269531 ], [ 0.0, 0.0, -1.3136863708496094 ] ]
    Implemented Gradient: [ [ 0.9800000190734863, 0.0, 0.0 ], [ 0.6039999723434448, 0.0, 0.0 ], [ -1.312000036239624, 0.0, 0.0 ], [ 0.0, 0.9800000190734863, 0.0 ], [ 0.0, 0.6039999723434448, 0.0 ], [ 0.0, -1.312000036239624, 0.0 ], [ 0.0, 0.0, 0.9800000190734863 ], [ 0.0, 0.0, 0.6039999723434448 ], [ 0.0, 0.0, -1.312000036239624 ] ]
    Error: [ [ -9.965896606445312E-5, 0.0, 0.0 ], [ -8.009672164916992E-4, 0.0, 0.0 ], [ -0.0016863346099853516, 0.0, 0.0 ], [ 0.0, -9.965896606445312E-5, 0.0 ], [ 0.0, -8.009672164916992E-4, 0.0 ], [ 0.0, -0.0016863346099853516, 0.0 ], [ 0.0, 0.0, -9.965896606445312E-5 ], [ 0.0, 0.0, -8.009672164916992E-4 ], [ 0.0, 0.0, -0.0016863346099853516 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.5796e-04 +- 6.4833e-04 [0.0000e+00 - 2.1040e-03] (36#)
    relativeTol: 4.5191e-04 +- 3.2924e-04 [4.5158e-05 - 1.0235e-03] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.22 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 7.4489 +- 1.3258 [6.4975 - 18.0848]
    Learning performance: 5.3469 +- 1.3444 [4.4172 - 14.2062]
    
```

