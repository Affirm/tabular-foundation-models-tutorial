export class Random {
  constructor(seed = 42) {
    this.mt = new Uint32Array(624);
    this.index = 624;
    this.seed(seed);
  }

  seed(seed) {
    if ((typeof seed !== "number" || !Number.isSafeInteger(seed)) && typeof seed !== "bigint") {
      throw new TypeError("Random seed must be a safe integer or bigint");
    }

    let value = typeof seed === "bigint" ? seed : BigInt(seed);
    if (value < 0n) value = -value;
    const key = [];
    do {
      key.push(Number(value & 0xffffffffn));
      value >>= 32n;
    } while (value > 0n);
    this.initByArray(key);
  }

  initGenRand(seed) {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const previous = this.mt[i - 1];
      this.mt[i] = (Math.imul(1812433253, previous ^ (previous >>> 30)) + i) >>> 0;
    }
    this.index = 624;
  }

  initByArray(key) {
    this.initGenRand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(624, key.length); k > 0; k--) {
      const previous = this.mt[i - 1];
      this.mt[i] = (
        (this.mt[i] ^ Math.imul(previous ^ (previous >>> 30), 1664525))
        + key[j]
        + j
      ) >>> 0;
      i++;
      j++;
      if (i >= 624) {
        this.mt[0] = this.mt[623];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (let k = 623; k > 0; k--) {
      const previous = this.mt[i - 1];
      this.mt[i] = (
        (this.mt[i] ^ Math.imul(previous ^ (previous >>> 30), 1566083941))
        - i
      ) >>> 0;
      i++;
      if (i >= 624) {
        this.mt[0] = this.mt[623];
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
    this.index = 624;
  }

  uint32() {
    if (this.index >= 624) this.twist();
    let value = this.mt[this.index++];
    value ^= value >>> 11;
    value ^= (value << 7) & 0x9d2c5680;
    value ^= (value << 15) & 0xefc60000;
    value ^= value >>> 18;
    return value >>> 0;
  }

  twist() {
    const mag = [0, 0x9908b0df];
    let value;
    let i = 0;
    for (; i < 227; i++) {
      value = (this.mt[i] & 0x80000000) | (this.mt[i + 1] & 0x7fffffff);
      this.mt[i] = this.mt[i + 397] ^ (value >>> 1) ^ mag[value & 1];
    }
    for (; i < 623; i++) {
      value = (this.mt[i] & 0x80000000) | (this.mt[i + 1] & 0x7fffffff);
      this.mt[i] = this.mt[i - 227] ^ (value >>> 1) ^ mag[value & 1];
    }
    value = (this.mt[623] & 0x80000000) | (this.mt[0] & 0x7fffffff);
    this.mt[623] = this.mt[396] ^ (value >>> 1) ^ mag[value & 1];
    this.index = 0;
  }

  random() {
    const a = this.uint32() >>> 5;
    const b = this.uint32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  next() {
    return this.random();
  }

  getrandbits(bits) {
    if (!Number.isInteger(bits) || bits < 0) throw new TypeError("Number of bits must be a non-negative integer");
    if (bits === 0) return 0;
    if (bits <= 32) return this.uint32() >>> (32 - bits);

    const words = Math.ceil(bits / 32);
    let remaining = bits;
    let result = 0n;
    for (let i = 0; i < words; i++) {
      let word = this.uint32();
      if (remaining < 32) word >>>= 32 - remaining;
      result |= BigInt(word) << BigInt(32 * i);
      remaining -= 32;
    }
    return bits <= 53 ? Number(result) : result;
  }

  randbelow(maxExclusive) {
    if (!Number.isSafeInteger(maxExclusive) || maxExclusive < 0) {
      throw new RangeError("Upper bound must be a non-negative safe integer");
    }
    if (maxExclusive === 0) return 0;
    const bits = BigInt(maxExclusive).toString(2).length;
    let value = this.getrandbits(bits);
    while (value >= maxExclusive) value = this.getrandbits(bits);
    return value;
  }

  int(maxExclusive) {
    return this.randbelow(maxExclusive);
  }

  choice(values) {
    return values[this.randbelow(values.length)];
  }

  shuffle(values) {
    const copy = values.slice();
    for (let i = copy.length - 1; i > 0; i--) {
      const j = this.randbelow(i + 1);
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }

  sample(values, count) {
    const n = values.length;
    if (!Number.isInteger(count) || count < 0 || count > n) {
      throw new RangeError("Sample larger than population or is negative");
    }

    const result = new Array(count);
    let setSize = 21;
    if (count > 5) setSize += 4 ** Math.ceil(Math.log(count * 3) / Math.log(4));
    if (n <= setSize) {
      const pool = values.slice();
      for (let i = 0; i < count; i++) {
        const j = this.randbelow(n - i);
        result[i] = pool[j];
        pool[j] = pool[n - i - 1];
      }
    } else {
      const selected = new Set();
      for (let i = 0; i < count; i++) {
        let j = this.randbelow(n);
        while (selected.has(j)) j = this.randbelow(n);
        selected.add(j);
        result[i] = values[j];
      }
    }
    return result;
  }
}
