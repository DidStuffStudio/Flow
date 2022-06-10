using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;

public class FlowField : MonoBehaviour
{
  public Vector3[] _vectors;
  private GameObject[] _arrows;
  private FlowParticle[] _particles;
  [SerializeField] private int width, height, depth;
  [SerializeField] private float spacing;
  [SerializeField] private GameObject arrow, particle;
  [SerializeField] private float rateOfChange = 100.0f;
  [SerializeField] private int particleCount = 100;
  [SerializeField] private float forceMultiplier = 100;
  [SerializeField] private float particleMass = 1;
  [SerializeField] private float particleDrag = 0.1f;
  [SerializeField] private float perlinDetail = 1.0f;
  public bool drawArrows;


  public int Width => width;

  public int Height => height;

  public int Depth => depth;

  public float Spacing => spacing;

  private void Start()
  {
    InstantiateGrid();
    InstantiateParticles();
  }



  private void Update()
  {
    UpdateVectors();
    UpdateParticles();
  }

  private void InstantiateGrid()
  {
    _vectors = new Vector3[Width*Height*Depth];
    _arrows = new GameObject[Width*Height*Depth];
    if (!drawArrows) return;
    for (int i = 0; i < Width; i++)
    {
      for (int j = 0; j < Height; j++)
      {
        for (int k = 0; k < Depth; k++)
        {
          
          var id = i + j*Width + k*Width*Height;
          var arrowGO = Instantiate(arrow);
          arrowGO.transform.position = new Vector3(i * Spacing, j * Spacing, k * Spacing);
          _arrows[id] = arrowGO;
        }
      }
    }
  }

  private void InstantiateParticles()
  {
    
    _particles = new FlowParticle[particleCount];
    for (int i = 0; i < particleCount; i++)
    {
      var p = Instantiate(particle);
      var pos = Vector3.zero;
      pos.x = Random.Range(0, width*spacing);
      pos.y = Random.Range(0, height*spacing);
      pos.z = Random.Range(0, depth*spacing);
      _particles[i] = p.GetComponent<FlowParticle>();
      _particles[i].Field = this;
      _particles[i].SetBounds();
      _particles[i].SetPosition(pos);
      /*_particles[i].transform.position = pos;
      _particles[i].CurrentPos = pos;
      _particles[i].AcceptNewPosition = true;*/
    }
  }
  
  private void UpdateVectors()
  {
    for (var i = 0; i < Width; i++)
    {
      for (var j = 0; j < Height; j++)
      {
        for (var k = 0; k < Depth; k++)
        {
          // blockIdx.x + blockIdx.y * gridDim.x  + gridDim.x * gridDim.y * blockIdx.z; 
          var id = i + j*Width + k*Width*Height;
          //_vectors[id] = PerlinNoise3D(i*perlinDetail, j*perlinDetail, k*perlinDetail, (Time.fixedTime*rateOfChange/100f));
          _vectors[id] = PerlinNoise3D(i*perlinDetail, j*perlinDetail, k*perlinDetail + (Time.fixedTime*rateOfChange/100f));
          if(drawArrows)_arrows[id].transform.LookAt(_vectors[id]);
        }
      }
    }
  }

  private void UpdateParticles()
  {
    for (var i = 0; i < particleCount; i++)
    {
      var pPos = _particles[i].CurrentPos;
      var x = Mathf.FloorToInt(pPos.x);
      var y = Mathf.FloorToInt(pPos.y);
      var z = Mathf.FloorToInt(pPos.z);

      var id = x + y * (Width / (int) spacing) + z * (Width / (int) spacing) * (Height / (int) spacing);
      _particles[i].SetPosition(CalculateNewPosition(_particles[i], _vectors[id].normalized * forceMultiplier));
    }
  }

  private Vector3 CalculateNewPosition(FlowParticle p, Vector3 flowAcceleration)
  {
    var currentPos = p.CurrentPos;
    var velocity = p.Velocity;
    var acceleration = p.Acceleration;
    var dt = Time.fixedDeltaTime;
    
    var newPos = currentPos + velocity*dt + acceleration*(dt*dt*0.5f);
    var newAcc = CalculateForces(velocity, flowAcceleration); // only needed if acceleration is not constant
    var newVel = velocity + (acceleration+newAcc)*(dt*0.5f);
    
    p.Velocity = newVel;
    p.Acceleration = newAcc;
    return newPos;
  }
  
  private Vector3 CalculateForces(Vector3 velocity, Vector3 flowAcceleration)
  {
    var absoluteVelocity = new Vector3(Mathf.Abs(velocity.x), Mathf.Abs(velocity.y), Mathf.Abs(velocity.z));
    var dragForce = 0.5f * particleDrag * Vector3.Scale(velocity,absoluteVelocity); // D = 0.5 * (rho * C * Area * vel^2)
    var dragAcc = dragForce / particleMass; // a = F/m
    return flowAcceleration - dragAcc;
  }



    private Vector3 PerlinNoise3D(float x, float y, float z)//, float w)
    {
      //X coordinate
      var xy = Mathf.PerlinNoise(x, y);
      var xz = Mathf.PerlinNoise(x, z);
      //var xw = Mathf.PerlinNoise(x, w);
 
 
      //Ycoordinate
      var yx = Mathf.PerlinNoise(y, x);
      var yz = Mathf.PerlinNoise(y, z);
      //var yw = Mathf.PerlinNoise(y, w);
 
 
      //Z coordinate
      var zx = Mathf.PerlinNoise(z, x);
      var zy = Mathf.PerlinNoise(z, y);
      //var zw = Mathf.PerlinNoise(z, w);
 
 
      /*//W coordinate
      var wx = Mathf.PerlinNoise(w, x);
      var wy = Mathf.PerlinNoise(w, y);
      var wz = Mathf.PerlinNoise(w, z);
      var outX = (xy + xz + xw + wx) / 4;
      var outY = (yx + yz + yw +wy) /4;
      var outZ = (zx + zy + zw + wz) /4;*/

      
      var outX = (xy + xz) / 2;
      var outY = (yx + yz) /2;
      var outZ = (zx + zy) /2;

      outX -= 0.5f;
      outY -= 0.5f;
      outZ -= 0.5f;

      outX *= 360;
      outY *= 360;
      outZ *= 360;
      
      return new Vector3(outX,outY,outZ);
    }

    public void SwitchParticle(FlowParticle oldP, FlowParticle newP)
    {
      var pList = _particles.ToList();
      var index = pList.IndexOf(oldP);
      _particles[index] = newP;
    }
  
}
