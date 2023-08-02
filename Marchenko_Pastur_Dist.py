def mp_pdf(sigma2, q, obs):
    lambda_plus = sigma2 * (1 + q ** 0.5) ** 2
    lambda_minus = sigma2 * (1 - q ** 0.5) ** 2
    l = np.linspace(lambda_minus, lambda_plus, obs)
    pdf_mp = 1 / (2 * np.pi * sigma2 * q * l) \
             * np.sqrt((lambda_plus - l)
                       * (l - lambda_minus))
    pdf_mp = pd.Series(pdf_mp, index=l)
    return pdf_mp


from sklearn.neighbors import KernelDensity


def kde_fit(bandwidth, obs, x=None):
    kde = KernelDensity(bandwidth, kernel='gaussian')
    if len(obs.shape) == 1:
        kde_fit = kde.fit(np.array(obs).reshape(-1, 1))
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logprob = kde_fit.score_samples(x)
    pdf_kde = pd.Series(np.exp(logprob), index=x.flatten())
    return pdf_kde


corr_mat = np.random.normal(size=(10000, 1000))
corr_coef = np.corrcoef(corr_mat, rowvar=0)
sigma2 = 1
obs = corr_mat.shape[0]
q = corr_mat.shape[0] / corr_mat.shape[1]


def plotting(corr_coef, q):
    ev, _ = np.linalg.eigh(corr_coef)
    idx = ev.argsort()[::-1]
    eigen_val = np.diagflat(ev[idx])
    pdf_mp = mp_pdf(1., q=corr_mat.shape[1] / corr_mat.shape[0],
                    obs=1000)
    kde_pdf = kde_fit(0.01, np.diag(eigen_val))
    ax = pdf_mp.plot(title="Marchenko-Pastur Theorem",
                     label="M-P", style='r--')
    kde_pdf.plot(label="Empirical Density", style='o-', alpha=0.3)
    ax.set(xlabel="Eigenvalue", ylabel="Frequency")
    ax.legend(loc="upper right")
    plt.show()
    return plt


plotting(corr_coef, q);
