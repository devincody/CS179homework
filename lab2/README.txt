Question 1.1:
The GK110 starts 2 instructions in up to 4 warps each clock. Since an 
arithmetic instruction takes 10 clock cycles to execute (10ns with a
1 GHz clock speed), we require 80 instructions in the queue to hide
the latency of a single instruction.


Question 1.2:
(a) No. The hardware threads are closely associated with threadIdx.x not
threadIdx.y. This means that all the threads in a given warp (threadIdx.x
 = 0...31) will evaluate idx % 32 as the same value. This means that any
given warp will evaluate only foo() or bar() and therefore will NOT
diverge.

(b) Yes. Consider threadIdx.x = 0 which doesnt even enter the for loop.
Conversely, all other threads will keep on running which means that these
threads have diverged. On the other hand, the warp scheduler doesn't have
to do anything special to tell the threads which have completed the for
loop to do nothing. So in this case, the only wasted resource comes from
the idling of the threads which have already finished. In this sense, the
code has not diverged.


Question 1.3:
(a) yes, because data access is memory aligned. Thread with threadIdx.x = 1
will read data that is immediately next to the data read by threadIDx.x=0.
This is basically to say that our stride here is 1. Therefore, we only 
need 32 cache lines per block.

(b) No. because the stride of this data access has a stride of blockSize.y.
This means that data is not coalesced. Consider two consecutive threads
with threadIdx.x = 1 and 2. Then the memory locations that they access are
separated by 31 places. This means that each thread in a warp will need a
seperate global memory access for a total of 32*32=1024 cache lines.

(c) Here the write is partially coalesced. The first 31 threads will be
able to write to global memory in one cache line, but the 32nd thread 
needs an additional cache line since it does not fit in the first access.
This is specifically because of the offset of 1 in the indexing. Finally,
this means that we need twice as many writes as the answer to Q1.3a. Which
is to say, 64 cache lines per block.


Question 1.4:
(a) No. output[] is written to with a stride of 1, which means there will
be no bank conflicts. lhs[] is read with a stride of 1, so again there are
no bank conflicts. rhs[] is trickier because each thread in a given warp
accesses the same value, but this is ok because the value will be broadcast
to all threads which doesn't constitute a bank conflict.

(b) 
1. load output[i+32*j] into register a
2. load lhs[i+32*k] into register b
3. load rhs[k+128*j] into register c
4. FMA with registers a,b,c (output to a)  //as defined in the HW instructions
5. store register a to output[i+32*j]

6. load output[i+32*j] into register a
7. load lhs[i+32*(k+1)] into register b
8. load rhs[(k+1)+128*j] into register c
9. FMA with registers a,b,c (output to a)  //as defined in the HW instructions
10. store register a to output[i+32*j]

(c)
step 4 is dependant on steps 1-3 being complete
step 5 is dependant on step 4
step 6 is dependant on step 5
step 9 is dependant on steps 6-8
step 10 is dependant on step 9

(d)

/* C CODE */
int i = threadIdx.x;
int j = threadIdx.y;

for (int k = 0; k < 128; k += 2) {
	l1 = lhs[i + 32 * k];
	l2 = lhs[i + 32 * (k + 1)];
	r1 = rhs[k + 128 * j];
	r2 = rhs[(k + 1) + 128 * j];

    output[i + 32 * j] += l1 * r1 + l2 * r2;
}

/* ASSEMBLY CODE */
1. load output[i+32*j] into register a
2. load lhs[i+32*k] into register b
3. load rhs[k+128*j] into register c
4. load lhs[i+32*(k+1)] into register d
5. load rhs[(k+1)+128*j] into register e

6. FMA with registers a,b,c (output to a)  //as defined in the HW instructions
7. FMA with registers a,d,e (output to a)  //as defined in the HW instructions

8. store register a to output[i+32*j]


(e) We could use a shared memory variable or register to hold a temporary 
value for the output and accumulate using that local variable and then
only at the end, write to global memory.


PART 2

NB: Sub optimialities commented in code

Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 10 seconds
Size 512 naive CPU: 0.671360 ms
Size 512 GPU memcpy: 0.023136 ms
Size 512 naive GPU: 0.046464 ms
Size 512 shmem GPU: 0.013696 ms
Size 512 optimal GPU: 0.010976 ms

Size 1024 naive CPU: 3.148736 ms
Size 1024 GPU memcpy: 0.048000 ms
Size 1024 naive GPU: 0.111264 ms
Size 1024 shmem GPU: 0.037248 ms
Size 1024 optimal GPU: 0.035520 ms

Size 2048 naive CPU: 37.409695 ms
Size 2048 GPU memcpy: 0.168928 ms
Size 2048 naive GPU: 0.389344 ms
Size 2048 shmem GPU: 0.148032 ms
Size 2048 optimal GPU: 0.136160 ms

Size 4096 naive CPU: 190.684097 ms
Size 4096 GPU memcpy: 0.540800 ms
Size 4096 naive GPU: 1.586272 ms
Size 4096 shmem GPU: 0.563520 ms
Size 4096 optimal GPU: 0.555840 ms




BONUS:
The naive code has two for loops which means the code will be slower because
of the loop overhead. Second, unless optimized by the compiler, vec_add will
read from global memory 4 times and write twice per index per call where as
the other code will read 3 times from global and write once.

TIME:
3 hours part 1
6 hours part 2

FEEDBACK:
Felt underequiped for this part 2 of the pset. Figuring out the correct indexing 
was unnecessarily difficult and I dont think I learned anything from doing it.
I think it would be immensely helpful for future years if students were given 
a visual depiction of the input, output, and shared matrices and then shown
geometrically how to calculate the requisite indices. It was a real pain 
trying to guess how each of the indices were calculated on top of learning
how row-major indexing worked and figuring how the for loops were working. 
The major concept here was global memory access and shared memory padding,
not learning how indexing works. Let me know if any of this was 
unclear, I'm happy to sit down to clarify.


