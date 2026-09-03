# TweetNaCl

`tweetnacl.c` and `tweetnacl.h` are from the TweetNaCl 2014-04-27 release:
<https://tweetnacl.cr.yp.to/software.html>.  They are public-domain software.

The imported C source SHA-256 is
`02e65bc3013ff2168983365e55906bc783c4c7e0a60d8100f17bb303a17175c4`.
Lana changes only the include path and uses `uint32_t` for TweetNaCl's required
32-bit `u32` type on LP64 platforms.  No cryptographic algorithm was changed.
