#define GROUP_SIZE_X 1024

struct cpx
{
    float x;
    float y;
};

cbuffer Constants
{
    float           angle;
    int             steps;
    unsigned int    lmask;
    int             dist;
    int             load_input;
};

StructuredBuffer<cpx> input;
RWStructuredBuffer<cpx> rwbuf_in;

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_fft(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = groupID.x * GROUP_SIZE_X + threadIDInGroup.x;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    cpx w;
    sincos(angle * ((tid << steps) & ((dist - 1) << steps)), w.y, w.x);
    cpx in_lower, in_upper;
    if (load_input != 0)
    {
        in_lower = input[in_low];
        in_upper = input[in_high];
    }
    else
    {
        in_lower = rwbuf_in[in_low];
        in_upper = rwbuf_in[in_high];
    }
    float x = in_lower.x - in_upper.x;
    float y = in_lower.y - in_upper.y;
    rwbuf_in[in_low].x = in_lower.x + in_upper.x;
    rwbuf_in[in_low].y = in_lower.y + in_upper.y;
    rwbuf_in[in_high].x = (w.x * x) - (w.y * y);
    rwbuf_in[in_high].y = (w.y * x) + (w.x * y);
}