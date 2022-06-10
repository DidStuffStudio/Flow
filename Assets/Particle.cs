using UnityEngine;

struct Particle
{
    public Vector3 Position { get; private set; }

    private Vector3 Velocity { get; set; }

    private Vector3 Acceleration { get; set; }

    public void Construct(Vector3 pos, Vector3 vel, Vector3 acc) // Constructor.
    {
        Position = pos;
        Velocity = vel;
        Acceleration = acc;
    }
}