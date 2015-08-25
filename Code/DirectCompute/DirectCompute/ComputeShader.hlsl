
ByteAddressBuffer Buffer0 : register(t0);
ByteAddressBuffer Buffer1 : register(t1);
RWByteAddressBuffer BufferOut : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int i0 = asint(Buffer0.Load(DTid.x * 8));
    float f0 = asfloat(Buffer0.Load(DTid.x * 8 + 4));
    int i1 = asint(Buffer1.Load(DTid.x * 8));
    float f1 = asfloat(Buffer1.Load(DTid.x * 8 + 4));

    int ir = 0;
    float fr = 0.0;
    for (int i = 0; i < 4096 / 8; ++i)
    {
        ir += i0 + i1;
        fr += f0 + f1;
        ir += i0 + i1;
        fr += f0 + f1;
        ir += i0 + i1;
        fr += f0 + f1;
        ir += i0 + i1;
        fr += f0 + f1;
        ir += i0 + i1;
        fr += f0 + f1;
        ir += i0 + i1;
        fr += f0 + f1;
        ir += i0 + i1;
        fr += f0 + f1;
        ir += i0 + i1;
        fr += f0 + f1;
    }        

    BufferOut.Store(DTid.x * 8, asuint(ir));
    BufferOut.Store(DTid.x * 8 + 4, asuint(fr));
}