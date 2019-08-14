#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include "fold.h"
#include "parameter.h"

bool allow_paired(char x, char y)
{
    x = std::tolower(x);
    y = std::tolower(y);
    return (x=='a' && y=='u') || (x=='u' && y=='a') || 
        (x=='c' && y=='g') || (x=='g' && y=='c') ||
        (x=='g' && y=='u') || (x=='u' && y=='g');
}

auto make_constraint(const std::string& seq, std::string stru, u_int32_t max_bp, bool canonical_only=true)
{
    const auto L = seq.size();
    if (stru.empty())
        stru = std::string(L, '.');
    std::vector<uint> bp(L+1, 0);
    std::stack<uint> st;
    for (auto i=0u; i!=stru.size(); ++i)
    {
        switch (stru[i])
        {
        case '(':
            st.push(i); break;
        case ')':
        {
            auto j=st.top();
            st.pop();
            bp[i+1] = j+1;
            bp[j+1] = i+1;
        }
        break;
        default: break;
        }
    }

    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1));
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired[i][i-1] = true; // the empty string is alway allowed to be unpaired
        allow_unpaired[i][i] = stru[i-1]=='.' || stru[i-1]=='x';
        bool bp_l = stru[i-1]=='.' || stru[i-1]=='<' || stru[i-1]=='|';
        for (auto j=i+1; j<=L; j++)
        {
            allow_paired[i][j] = j-i > max_bp;
            bool bp_r = stru[j-1]=='.' || stru[j-1]=='>' || stru[j-1]=='|';
            allow_paired[i][j] = allow_paired[i][j] && ((bp_l && bp_r) || bp[i]==j);
            if (canonical_only)
                allow_paired[i][j] = allow_paired[i][j] && ::allow_paired(seq[i-1], seq[j-1]);
            allow_unpaired[i][j] = allow_unpaired[i][j-1] && allow_unpaired[j][j];
        }
    }
    return std::make_pair(allow_paired, allow_unpaired);
}

Fold::
Fold(std::unique_ptr<MFETorch>&& p, size_t min_hairpin_loop_length, size_t max_internal_loop_length)
    :   param(std::move(p)), 
        min_hairpin_loop_length_(min_hairpin_loop_length),
        max_internal_loop_length_(max_internal_loop_length)
{

}

bool
Fold::
update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int32_t k)
{
    if (max_v.item<float>() < new_v.item<float>()) 
    {
        max_v = new_v;
        max_t = {tt, k};
        return true;
    }
    return false;
}

bool 
Fold::
update_max(ScoreType& max_v, ScoreType new_v, TB& max_t, TBType tt, u_int8_t p, u_int8_t q)
{
    if (max_v.item<float>() < new_v.item<float>()) 
    {
        max_v = new_v;
        max_t = {tt, std::make_pair(p, q)};
        return true;
    }
    return false;
}

auto 
Fold::
compute_viterbi(const std::string& seq, const std::string& stru) -> ScoreType
{
    const auto seq2 = param->convert_sequence(seq);
    const auto L = seq.size();
    const ScoreType NEG_INF = torch::full({}, std::numeric_limits<float>::lowest(), torch::requires_grad(false));
    Cv_.clear();  Cv_.resize(L+1, VI(L+1, NEG_INF));
    Mv_.clear();  Mv_.resize(L+1, VI(L+1, NEG_INF));
    M1v_.clear(); M1v_.resize(L+1, VI(L+1, NEG_INF));
    Fv_.clear();  Fv_.resize(L+1, NEG_INF);
    Ct_.clear();  Ct_.resize(L+1, VT(L+1));
    Mt_.clear();  Mt_.resize(L+1, VT(L+1));
    M1t_.clear(); M1t_.resize(L+1, VT(L+1));
    Ft_.clear();  Ft_.resize(L+1);

    const auto [allow_paired, allow_unpaired] = make_constraint(seq, stru, min_hairpin_loop_length_);

    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (allow_paired[i][j])
            {
                if (allow_unpaired[i+1][j-1])
                    update_max(Cv_[i][j], param->hairpin(seq2, i, j), Ct_[i][j], TBType::C_HAIRPIN_LOOP);

                for (auto k=i+1; (k-1)-(i+1)+1<max_internal_loop_length_ && k<j; k++)
                    if (allow_unpaired[i+1][k-1])
                        for (auto l=j-1; k<l && ((k-1)-(i+1)+1)+((j-1)-(l+1)+1)<max_internal_loop_length_; l--)
                            if (allow_paired[k][l] && allow_unpaired[l+1][j-1])
                                update_max(Cv_[i][j], Cv_[k][l] + param->single_loop(seq2, i, j, k, l), Ct_[i][j], TBType::C_INTERNAL_LOOP, k, l);

                for (auto u=i+1; u+1<=j-1; u++)
                    update_max(Cv_[i][j], Mv_[i+1][u]+M1v_[u+1][j-1] + param->multi_loop(seq2, i, j), Ct_[i][j], TBType::C_MULTI_LOOP, u);

            }

            /////////////////
            if (allow_paired[i][j])
                update_max(Mv_[i][j], Cv_[i][j] + param->multi_paired(seq2, i, j), Mt_[i][j], TBType::M_PAIRED, i);

            ScoreType t = torch::zeros({}, torch::dtype(torch::kFloat));
            for (auto u=i; u+1<j; u++)
            {
                if (!allow_unpaired[u][u]) break;
                t += param->multi_unpaired(seq2, u);
                if (allow_paired[u+1][j])
                {
                    auto s = param->multi_paired(seq2, u+1, j) + t;
                    update_max(Mv_[i][j], Cv_[u+1][j]+s, Mt_[i][j], TBType::M_PAIRED, u+1);
                }
            }

            for (auto u=i; u+1<=j; u++)
                if (allow_paired[u+1][j])
                    update_max(Mv_[i][j], Mv_[i][u]+Cv_[u+1][j] + param->multi_paired(seq2, u+1, j), Mt_[i][j], TBType::M_BIFURCATION, u);

            if (allow_unpaired[j][j])
                update_max(Mv_[i][j], Mv_[i][j-1] + param->multi_unpaired(seq2, j), Mt_[i][j], TBType::M_UNPAIRED);

            /////////////////
            if (allow_paired[i][j])
                update_max(M1v_[i][j], Cv_[i][j] + param->multi_paired(seq2, i, j), M1t_[i][j], TBType::M1_PAIRED);

            if (allow_unpaired[j][j])
                update_max(M1v_[i][j], M1v_[i][j-1] + param->multi_unpaired(seq2, j), M1t_[i][j], TBType::M1_UNPAIRED);
        }
    }

    update_max(Fv_[L], param->external_zero(seq2), Ft_[L], TBType::F_ZERO);

    for (auto i=L-1; i>=1; i--)
    {
        if (allow_unpaired[i][i])
            update_max(Fv_[i], Fv_[i+1] + param->external_unpaired(seq2, i), Ft_[i], TBType::F_UNPAIRED);

        for (auto k=i+1; k+1<=L; k++)
            if (allow_paired[i][k])
                update_max(Fv_[i], Cv_[i][k]+Fv_[k+1] + param->external_paired(seq2, i, k), Ft_[i], TBType::F_BIFURCATION, k);
    }

    update_max(Fv_[1], Cv_[1][L] + param->external_paired(seq2, 1, L), Ft_[1], TBType::F_PAIRED);

    return Fv_[1];
}

auto
Fold::
traceback_viterbi() -> std::vector<u_int32_t>
{
    const auto L = Ft_.size()-1;
    std::vector<u_int32_t> pair(L+1, 0);
    std::queue<std::tuple<TB, u_int32_t, u_int32_t>> tb_queue;
    tb_queue.emplace(Ft_[1], 1, L);

    while (!tb_queue.empty())
    {
        const auto [tb, i, j] = tb_queue.front();
        const auto [tb_type, kl] = tb;
        tb_queue.pop();

        switch (tb_type)
        {
            case TBType::C_HAIRPIN_LOOP: {
                break;
            }
            case TBType::C_INTERNAL_LOOP: {
                const auto [p, q] = std::get<1>(kl);
                pair[p] = q;
                pair[q] = p;
                tb_queue.emplace(Ct_[p][q], p, q);
                break;
            }
            case TBType::C_MULTI_LOOP: {
                const auto k = std::get<0>(kl);
                tb_queue.emplace(Mt_[i+1][k], i+1, k);
                tb_queue.emplace(M1t_[k+1][j-1], k+1, j-1);
                break;
            }
            case TBType::M_PAIRED: {
                const auto k = std::get<0>(kl);
                pair[k] = j;
                pair[j] = k;
                tb_queue.emplace(Ct_[k][j], k, j);
                break;
            }
            case TBType::M_BIFURCATION: {
                const auto k = std::get<0>(kl);
                pair[k+1] = j;
                pair[j] = k+1;
                tb_queue.emplace(Mt_[i][k], i, k);
                tb_queue.emplace(Ct_[k+1][j], k+1, j);
                break;
            }
            case TBType::M_UNPAIRED: {
                tb_queue.emplace(Mt_[i][j-1], i, j-1);
                break;
            }    
            case TBType::M1_PAIRED: {
                pair[i] = j;
                pair[j] = i;
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
            case TBType::M1_UNPAIRED: {
                tb_queue.emplace(M1t_[i][j-1], i, j-1);
                break;
            }
            case TBType::F_ZERO: {
                break;
            }
            case TBType::F_UNPAIRED: {
                tb_queue.emplace(Ft_[i+1], i+1, j);
                break;
            }
            case TBType::F_BIFURCATION: {
                const auto k = std::get<0>(kl);
                pair[i] = k;
                pair[k] = i;
                tb_queue.emplace(Ct_[i][k], i, k);
                tb_queue.emplace(Ft_[k+1], k+1, j);
                break;
            }
            case TBType::F_PAIRED: {
                pair[i] = j;
                pair[j] = i;
                tb_queue.emplace(Ct_[i][j], i, j);
                break;
            }
        }
    }

    return pair;
}

#if 0
auto nussinov(const std::string& seq)
{
    const auto L = seq.size();
    std::vector dp(L+1, std::vector(L+1, 0));
    for (auto i=L; i>=1; i--)
    {
        for (auto j=i+1; j<=L; j++)
        {
            if (j-i>3 && allow_paired(seq[i-1], seq[j-1]))
                dp[i][j] = std::max(dp[i][j], dp[i+1][j-1]+1);
            dp[i][j] = std::max(dp[i][j], dp[i+1][j]);
            dp[i][j] = std::max(dp[i][j], dp[i][j-1]);  
            for (auto k=i+1; k<j-1; k++)
                dp[i][j]  = std::max(dp[i][j], dp[i][k]+dp[k+1][j]);
        }
    }

    return dp[1][L];
}
#endif