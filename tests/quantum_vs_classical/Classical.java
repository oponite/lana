/** Classical probability model: a single scalar p, no disposition.
 *
 *  Mirrors classical.py. Runs the same five experiments but can only see p,
 *  so it cannot distinguish states that differ only in phase (disposition d).
 */
public class Classical {
    static double append(double pa, double pb) { return 1.0 - (1.0 - pa) * (1.0 - pb); }
    static double invert(double p) { return 1.0 - p; }
    static double distance(double a, double b) { return Math.abs(a - b); }

    public static void main(String[] args) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"distinguish\":{\"classical\":{\"plus\":0.5,\"minus\":0.5}},");
        sb.append("\"invert\":{\"classical\":{\"before\":0.3,\"after\":")
          .append(invert(0.3)).append("}},");
        sb.append("\"distance\":{\"classical\":").append(distance(0.5, 0.5)).append("},");
        sb.append("\"neutralize\":{\"classical\":{\"before\":0.5,\"after\":0.5}},");
        sb.append("\"agreement\":{\"classical\":{\"agree\":").append(append(0.7, 0.7))
          .append(",\"oppose\":").append(append(0.7, 0.7)).append("}}}");
        System.out.println(sb);
    }
}
