using System.Linq;
using UnityEngine;


public static class FlowComputeShaderProperties
{
  public const string PositionBufferName       = "position_buffer";
  public const string VelocityBufferName       = "velocity_buffer";
  public const string AccelerationBufferName       = "acceleration_buffer";
  public const string PropertiesBufferName     = "properties_buffer";
  public const string ForcesBuffer      = "forces_buffer";
  public const string PerlinBuffer      = "perlin_buffer";
  public const int    NumProperties            = 19; // NOTE --> Change whenever we add a new field to the properties buffer.
  public const string ForcesKernel    = "cs_main_forces";
  public const string ParticleKernel    = "cs_main_particles";
}

public class FlowFieldGPU : MonoBehaviour
{ 
  
  private Vector3[] _forces;
  private GameObject[] _arrows;
  private FlowParticleGPU[] _particles;
  [SerializeField] private GameObject arrow, particle;
  [SerializeField] private int particleCount = 512;
  [SerializeField] private ComputeShader flowShader;
  
  [Header("Perlin Noise Parameters")]
  [SerializeField] private float offsetX = 0.0f;
  [SerializeField] private float offsetY = 0.0f;      
  [SerializeField] private int octaves = 7;
  [SerializeField] private float lacunarity = 2f;
  [SerializeField] private float gain = 0.5f;
  [SerializeField] private float  amplitude = 1.5f;
  [SerializeField] private float  frequency = 2.0f;
  [SerializeField] private float  power = 1.0f;
  [SerializeField] private float  scale = 1.0f;

  [Header("Misc Parameters")]
  [SerializeField] private float rateOfChange = 10;
  [SerializeField] private int width = 8;
  [SerializeField] private int height = 8;
  [SerializeField] private int depth = 8;
  [SerializeField] private int spacing = 1;
  
  [Header("Physics")]
  [SerializeField] private float particleDrag = 0.1f;
  [SerializeField] private float particleMass = 1.0f;
  [SerializeField] private float forceMultiplier = 10;
  [SerializeField] private float speed = 1;
  
  private ComputeBuffer _forcesBuffer;
  private ComputeBuffer _positionBuffer;
  private ComputeBuffer _velocityBuffer;
  private ComputeBuffer _accelerationBuffer;
  private ComputeBuffer _propertiesBuffer;
  private ComputeBuffer _perlinBuffer;
  
  private const int GridUnitSideX       = 1;
  private const int GridUnitSideY       = 1;
  private const int GridUnitSideZ       = 1;
  
  private const int NumThreadsPerGroupX = 8;
  private const int NumThreadsPerGroupY = 8;
  private const int NumThreadsPerGroupZ = 8;
  
  private int _gridResX;
  private int _gridResY;
  private int _gridResZ;
  
  private int _vertCount;
  
  private int _forcesKernel;
  private int _particleKernel;

  public int Width => width;

  public int Height => height;

  public int Depth => depth;

  public float Spacing => spacing;

  private void Start()
  {
    Initialise();
  }

  private void Initialise()
  { 
    _gridResX  = GridUnitSideX * NumThreadsPerGroupX;
    _gridResY  = GridUnitSideY * NumThreadsPerGroupY;
    _gridResZ  = GridUnitSideZ * NumThreadsPerGroupZ;
    _vertCount = _gridResX * _gridResY * _gridResZ;
    CreateBuffers();
    InstantiateGrid();
    InstantiateParticles();
  }

  private void CreateBuffers()
  {
    var sizeOfThreeFloat = sizeof(float) * 3;
    _positionBuffer       = new ComputeBuffer (particleCount, sizeOfThreeFloat);
    _velocityBuffer       = new ComputeBuffer (particleCount, sizeOfThreeFloat);
    _accelerationBuffer       = new ComputeBuffer (particleCount, sizeOfThreeFloat);
    _propertiesBuffer     = new ComputeBuffer (FlowComputeShaderProperties.NumProperties, sizeof (float));
    _forcesBuffer = new ComputeBuffer(_vertCount, sizeOfThreeFloat);
    _perlinBuffer = new ComputeBuffer(_vertCount, sizeof(float));
    ResetBuffers();
    
    _forcesKernel = flowShader.FindKernel (FlowComputeShaderProperties.ForcesKernel);
    _particleKernel = flowShader.FindKernel (FlowComputeShaderProperties.ParticleKernel);
  }

  private void Update()
  {
    Dispatch();
    offsetX += Time.fixedDeltaTime * speed / 100;
    offsetY += Time.fixedDeltaTime * speed / 100;
    UpdateVectors();
    UpdateParticles();
  }

  private void SetUpPropertiesBuffer()
  {
    _propertiesBuffer.SetData (new[]
    {
      offsetX,offsetY,octaves,lacunarity,gain, amplitude, frequency,
      power, scale, Time.fixedTime, Time.fixedDeltaTime, rateOfChange,
      width, height, depth, spacing, particleDrag, particleMass, forceMultiplier
    } );
  }

  private void SetUpForcesKernel()
  {
    flowShader.SetBuffer (_forcesKernel, FlowComputeShaderProperties.PropertiesBufferName, _propertiesBuffer);
    flowShader.SetBuffer (_forcesKernel, FlowComputeShaderProperties.PositionBufferName, _positionBuffer);
    flowShader.SetBuffer (_forcesKernel, FlowComputeShaderProperties.VelocityBufferName, _velocityBuffer);
    flowShader.SetBuffer (_forcesKernel, FlowComputeShaderProperties.AccelerationBufferName, _accelerationBuffer);
    flowShader.SetBuffer (_forcesKernel, FlowComputeShaderProperties.ForcesBuffer, _forcesBuffer);
    flowShader.SetBuffer (_forcesKernel, FlowComputeShaderProperties.PerlinBuffer, _perlinBuffer);
  }  
  
  private void SetUpParticlesKernel()
  {
    flowShader.SetBuffer (_particleKernel, FlowComputeShaderProperties.PropertiesBufferName, _propertiesBuffer);
    flowShader.SetBuffer (_particleKernel, FlowComputeShaderProperties.PositionBufferName, _positionBuffer);
    flowShader.SetBuffer (_particleKernel, FlowComputeShaderProperties.VelocityBufferName, _velocityBuffer);
    flowShader.SetBuffer (_particleKernel, FlowComputeShaderProperties.AccelerationBufferName, _accelerationBuffer);
    flowShader.SetBuffer (_particleKernel, FlowComputeShaderProperties.ForcesBuffer, _forcesBuffer);
    flowShader.SetBuffer (_particleKernel, FlowComputeShaderProperties.PerlinBuffer, _perlinBuffer);
  }
  
  private void Dispatch()
  {
    SetUpPropertiesBuffer();
    SetUpForcesKernel();
    SetUpParticlesKernel();
    flowShader.Dispatch(_forcesKernel, GridUnitSideX, GridUnitSideY, GridUnitSideZ);
    flowShader.Dispatch(_particleKernel, GridUnitSideX, 1, 1);
  }
  
  private void InstantiateGrid()
  {
    _forces = new Vector3[Width*Height*Depth];
    _arrows = new GameObject[Width*Height*Depth];
    for (var i = 0; i < Width; i++)
    {
      for (var j = 0; j < Height; j++)
      {
        for (var k = 0; k < Depth; k++)
        {
          var arrowGO = Instantiate(arrow);
          arrowGO.transform.position = new Vector3(i * Spacing, j * Spacing, k * Spacing);
          var id = i + j*Width + k*Width*Height;
          _arrows[id] = arrowGO;
        }
      }
    }
  }

  private void InstantiateParticles()
  {
    
    _particles = new FlowParticleGPU[particleCount];
    for (var i = 0; i < particleCount; i++)
    {
      var p = Instantiate(particle);
      var pos = Vector3.zero;
      pos.x = Random.Range(0, width*spacing);
      pos.y = Random.Range(0, height*spacing);
      pos.z = Random.Range(0, depth*spacing);
      _particles[i] = p.GetComponent<FlowParticleGPU>();
      _particles[i].Field = this;
      _particles[i].SetBounds();
      _particles[i].SetPosition(pos);
    }
  }

  private void UpdateVectors()
  {
    var vecs = new Vector3[_vertCount];
    _forcesBuffer.GetData(vecs);
    
    for (var i = 0; i < Width; i++)
    {
      for (var j = 0; j < Height; j++)
      {
        for (var k = 0; k < Depth; k++)
        {
          var id = i + j*Width + k*Width*Height;
          _arrows[id].transform.rotation = Quaternion.Euler((vecs[id])*180);
        }
      }
    }
  }

  private void UpdateParticles()
  {
    var particleInfo = new Vector3[particleCount];
    _positionBuffer.GetData(particleInfo);
    for (var i = 0; i < particleCount; i++)
    {
      _particles[i].SetPosition(particleInfo[i]);
      //print("particle info: "+particleInfo[i]);
    }
  }

  private void ResetBuffers()
  {
    var particles = new Vector3[particleCount];
    var forces = new Vector3[_vertCount];
    var velocities = new Vector3[particleCount];
    
    for (float i = 0; i < particleCount; i += 1)
    {
      var x = Random.Range(0, width * spacing);
      var y = Random.Range(0, height * spacing);
      var z = Random.Range(0, depth * spacing);
      
      /*var xv = Random.Range(0, 1);
      var yv = Random.Range(0, 1);
      var zv = Random.Range(0, 1);*/
      particles[(int) i] = new Vector3(x,y,z);
      //velocities[(int) i] = new Vector3(xv,yv,zv);
    }
    _positionBuffer.SetData(particles);
    _velocityBuffer.SetData(velocities);
    _accelerationBuffer.SetData(velocities);
    _forcesBuffer.SetData(forces);
    _perlinBuffer.SetData(new float[_vertCount]);
  }

  private void ReleaseBuffers()
    {
      _forcesBuffer?.Release();
      _positionBuffer?.Release();
      _velocityBuffer?.Release();
      _accelerationBuffer?.Release();
      _propertiesBuffer?.Release();
      _perlinBuffer?.Release();
    }
  
  public void SwitchParticle(FlowParticleGPU oldP, FlowParticleGPU newP)
  {
    var newVels = new Vector3[particleCount];
    var newAccels = new Vector3[particleCount];
    _velocityBuffer.GetData(newVels);
    _accelerationBuffer.GetData(newAccels);
    
    var pList = _particles.ToList();
    var index = pList.IndexOf(oldP);
    newVels[pList.IndexOf(oldP)] = newP.Velocity;
    newAccels[pList.IndexOf(oldP)] = newP.Acceleration;
    _particles[index] = newP;
    
    _velocityBuffer.SetData(newVels);
    _accelerationBuffer.SetData(newAccels);
  }

    private void OnDisable()
    {
      ReleaseBuffers();
    }
  
}
